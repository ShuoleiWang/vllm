# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abort/update/resubmit for one sequence on a dedicated AsyncLLM.

Set the initial weight label before calling this helper. ``transfer_weights``
loads the new target weights; this helper owns start/finish and pause/resume.
Use ``save_partial`` to persist the prefix before a potentially failing update.
Sampling parameters are preserved, except for cumulative output and requesting
sampled-token logprobs. Use token-local sampling and token stops; other stateful
features need their own continuation protocol. The returned attempts retain
optional sampling masks and routed experts in their original request coordinates.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.outputs import CompletionOutput, WeightVersionSpan
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM


@dataclass
class Rollout:
    token_ids: list[int]
    sampled_token_logprobs: list[float]
    weight_versions: list[WeightVersionSpan]
    attempts: list[CompletionOutput]
    logprobs_mode: str

    @property
    def finish_reason(self) -> str | None:
        return self.attempts[-1].finish_reason if self.attempts else None

    @property
    def stop_reason(self) -> int | str | None:
        return self.attempts[-1].stop_reason if self.attempts else None


def join_attempts(
    attempts: Iterable[CompletionOutput], *, logprobs_mode: str
) -> Rollout:
    """Join terminal attempts; combine prior ``rollout.attempts`` for more cycles.

    Original logprobs are retained, not recomputed under the new weights. Replay
    metadata stays on each attempt because its coordinates may include the prompt.
    All attempts must use the supplied engine logprobs mode; raw and processed
    logprobs describe different distributions and must not be mixed.
    """
    if logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
        raise ValueError("Expected raw_logprobs or processed_logprobs, not logits")
    result = Rollout([], [], [], list(attempts), logprobs_mode)
    for attempt in result.attempts:
        if not attempt.finished():
            raise ValueError("Only terminal, cumulative attempts can be joined")
        offset = len(result.token_ids)
        if attempt.weight_versions is None:
            raise ValueError("The attempt ended without weight-version metadata")
        end = 0
        for span in attempt.weight_versions:
            if span.start != end or not span.start < span.end <= len(attempt.token_ids):
                raise ValueError("Weight-version spans do not cover the attempt")
            end = span.end
            shifted = WeightVersionSpan(span.version, offset + span.start, offset + end)
            if (
                result.weight_versions
                and result.weight_versions[-1].version == span.version
            ):
                result.weight_versions[-1].end = shifted.end
            else:
                result.weight_versions.append(shifted)
        if end != len(attempt.token_ids):
            raise ValueError("Weight-version spans do not cover every output token")
        result.sampled_token_logprobs.extend(
            logprobs[token].logprob
            for token, logprobs in zip(
                attempt.token_ids, attempt.logprobs or [], strict=True
            )
        )
        result.token_ids.extend(attempt.token_ids)
    return result


def _validate_sampling_params(params: SamplingParams) -> None:
    if params.n != 1:
        raise ValueError("Use one controller per sequence (n=1)")
    if (
        params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.repetition_penalty != 1
        or params.structured_outputs is not None
        or params.stop
        or params.bad_words
        or params.repetition_detection is not None
    ):
        raise ValueError(
            "Resubmission cannot restore penalties, text constraints, repetition "
            "detection or structured decoding"
        )
    if params.prompt_logprob_token_ids is not None:
        raise ValueError("Prompt scoring coordinates change when resubmitting")


async def abort_and_resume_rollout(
    engine: "AsyncLLM",
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    transfer_weights: Callable[[], Awaitable[None]],
    new_version: str,
    *,
    abort_after: int = 32,
    save_partial: Callable[[Rollout], Awaitable[None]] | None = None,
) -> Rollout:
    """Collect a prefix, save it, update, then use the remaining token budget.

    ``abort_after`` triggers a pause after a delivered output reaches the threshold;
    the final prefix can be longer. Natural completion before the pause skips the
    update. Save/update failures and cancellation leave a paused engine paused.
    A supplied seed starts a fresh RNG stream on resubmission; exact uninterrupted
    sampling equivalence is not promised. Coordinate access to this engine outside
    this helper: pausing aborts all of its requests, not just this sequence.
    """
    _validate_sampling_params(sampling_params)
    logprobs_mode = engine.model_config.logprobs_mode
    if logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
        raise ValueError("Configure the engine to return logprobs, not logits")
    max_tokens = sampling_params.max_tokens
    if max_tokens is None or not 0 < abort_after < max_tokens:
        raise ValueError("Require a finite max_tokens and 0 < abort_after < max_tokens")
    if new_version == "default":
        raise ValueError("The new checkpoint must have a known weight-version label")
    if await engine.get_weight_version() == "default":
        raise ValueError("Set the initial checkpoint label before starting rollouts")
    ready = asyncio.Event()

    async def collect(
        prompt: list[int], budget: int, signal_ready: bool = False
    ) -> CompletionOutput:
        final = None
        # Deep-copy even when skip_clone is set: this helper reuses the caller's
        # configuration for a second request with a different generation budget.
        params = deepcopy(sampling_params)
        params.max_tokens = budget
        params.min_tokens = max(0, sampling_params.min_tokens - (max_tokens - budget))
        params.output_kind = RequestOutputKind.CUMULATIVE
        if params.logprobs is None:
            params.logprobs = 0
        try:
            async for output in engine.generate(
                {"prompt_token_ids": prompt}, params, request_id=random_uuid()
            ):
                final = output.outputs[0]
                if signal_ready and len(final.token_ids) >= abort_after:
                    ready.set()
        finally:
            if signal_ready:
                ready.set()
        if final is None or not final.finished():
            raise RuntimeError("Generation ended without a terminal output")
        return final

    first_task = asyncio.create_task(collect(prompt_token_ids, max_tokens, True))
    try:
        await ready.wait()
        if first_task.done():
            return join_attempts([await first_task], logprobs_mode=logprobs_mode)
        await engine.pause_generation(mode="abort", clear_cache=True)
        # Pause completion alone does not mean the frontend delivered the result.
        first = await first_task
    except BaseException:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
        raise

    partial = join_attempts([first], logprobs_mode=logprobs_mode)
    if save_partial is not None:
        await save_partial(partial)
    await engine.start_weight_update()
    await transfer_weights()
    await engine.finish_weight_update(weight_version=new_version)
    # A failed transfer/finish leaves the engine paused for controller recovery.
    await engine.resume_generation()

    remaining = max_tokens - len(first.token_ids)
    if first.finish_reason != "abort" or remaining <= 0:
        return partial
    second = await collect(prompt_token_ids + list(first.token_ids), remaining)
    return join_attempts([first, second], logprobs_mode=logprobs_mode)
