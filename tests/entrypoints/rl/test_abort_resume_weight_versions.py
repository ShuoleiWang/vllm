# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for controller ordering and retained rollout data, without a model."""

import asyncio
from types import SimpleNamespace

import pytest

from examples.rl.abort_resume_weight_versions import (
    abort_and_resume_rollout,
    join_attempts,
)
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.outputs import CompletionOutput, SamplingMask, WeightVersionSpan
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)

pytestmark = pytest.mark.skip_global_cleanup


def completion(ids, version="A", reason="abort"):
    return CompletionOutput(
        index=0,
        text="",
        token_ids=list(ids),
        logprobs=[{token: Logprob(logprob=-token / 100)} for token in ids],
        cumulative_logprob=None,
        finish_reason=reason,
        weight_versions=[WeightVersionSpan(version, 0, len(ids))] if ids else [],
    )


class FakeEngine:
    def __init__(self, *, natural=False, failure=None, reason="abort"):
        self.model_config = SimpleNamespace(logprobs_mode="raw_logprobs")
        self.natural, self.failure, self.reason = natural, failure, reason
        self.paused = False
        self.pause_seen = asyncio.Event()
        self.release_terminal = asyncio.Event()
        self.release_terminal.set()
        self.closed = asyncio.Event()
        self.events, self.calls = [], []

    async def get_weight_version(self):
        return "A"

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))
        if len(self.calls) == 2:
            assert not self.paused
            yield SimpleNamespace(outputs=[completion([40, 50, 60], "B", "length")])
            return
        try:
            if self.natural:
                yield SimpleNamespace(outputs=[completion([10], reason="stop")])
                return
            yield SimpleNamespace(outputs=[completion([10, 20], reason=None)])
            await self.pause_seen.wait()
            await self.release_terminal.wait()
            self.events.append("terminal")
            yield SimpleNamespace(
                outputs=[completion([10, 20, 30], reason=self.reason)]
            )
        finally:
            self.closed.set()

    async def pause_generation(self, mode, clear_cache):
        assert mode == "abort" and clear_cache
        self.paused = True
        self.pause_seen.set()

    def record(self, operation):
        assert self.paused and self.closed.is_set()
        self.events.append(operation)
        if operation == self.failure:
            raise RuntimeError(operation)

    async def start_weight_update(self):
        self.record("start")

    async def transfer(self):
        self.record("transfer")

    async def finish_weight_update(self, weight_version):
        assert weight_version == "B"
        self.record("finish")

    async def resume_generation(self):
        self.record("resume")
        self.paused = False


async def run(engine, params=None, save=None):
    return await abort_and_resume_rollout(
        engine,
        [1, 2],
        params if params is not None else SamplingParams(max_tokens=6),
        engine.transfer,
        "B",
        abort_after=2,
        save_partial=save,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
async def test_wait_save_update_resubmit_preserves_sampling_and_original_logprobs(mode):
    engine = FakeEngine()
    engine.model_config.logprobs_mode = mode
    engine.release_terminal.clear()
    saved = []

    async def save(partial):
        engine.record("save")
        saved.append(partial)

    params = SamplingParams(
        max_tokens=6,
        min_tokens=5,
        temperature=0.7,
        top_p=0.9,
        seed=17,
        stop_token_ids=[99],
        logprobs=2,
        output_kind=RequestOutputKind.DELTA,
        skip_clone=True,
    )
    task = asyncio.create_task(run(engine, params, save))
    await asyncio.wait_for(engine.pause_seen.wait(), 1)
    assert not task.done() and engine.events == []
    engine.release_terminal.set()
    result = await asyncio.wait_for(task, 1)

    assert engine.events == [
        "terminal",
        "save",
        "start",
        "transfer",
        "finish",
        "resume",
    ]
    assert result.token_ids == [10, 20, 30, 40, 50, 60]
    assert result.sampled_token_logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    assert result.logprobs_mode == saved[0].logprobs_mode == mode
    assert result.weight_versions == [
        WeightVersionSpan("A", 0, 3),
        WeightVersionSpan("B", 3, 6),
    ]
    assert result.finish_reason == "length" and saved[0].finish_reason == "abort"
    assert saved[0].token_ids == [10, 20, 30]
    first, second = engine.calls
    assert second[0] == {"prompt_token_ids": [1, 2, 10, 20, 30]}
    assert first[2] != second[2]
    assert (second[1].max_tokens, second[1].min_tokens) == (3, 2)
    for _, actual, _ in engine.calls:
        assert (actual.temperature, actual.top_p, actual.seed) == (0.7, 0.9, 17)
        assert actual.logprobs == 2 and actual.stop_token_ids == [99]
        assert actual.output_kind == RequestOutputKind.CUMULATIVE
        assert actual.stop_token_ids is not params.stop_token_ids
    assert params.output_kind == RequestOutputKind.DELTA and params.max_tokens == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["save", "start", "transfer", "finish"])
async def test_update_failure_keeps_saved_prefix_and_engine_paused(failure):
    engine = FakeEngine(failure=failure)
    saved = []

    async def save(partial):
        saved.append(partial)
        engine.record("save")

    with pytest.raises(RuntimeError, match=failure):
        await run(engine, save=save)
    assert engine.paused and "resume" not in engine.events and len(engine.calls) == 1
    assert engine.events[-1] == failure
    assert saved[0].sampled_token_logprobs == [-0.1, -0.2, -0.3]
    assert saved[0].weight_versions == [WeightVersionSpan("A", 0, 3)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "natural,reason,budget",
    [(True, "stop", 6), (False, "stop", 6), (False, "abort", 3)],
)
async def test_completion_or_exhausted_budget_does_not_resubmit(
    natural, reason, budget
):
    engine = FakeEngine(natural=natural, reason=reason)
    result = await run(engine, SamplingParams(max_tokens=budget))
    assert len(engine.calls) == 1 and not engine.paused
    assert result.finish_reason == ("stop" if natural else reason)
    if natural:
        assert engine.events == []


@pytest.mark.asyncio
async def test_cancellation_closes_collector_and_does_not_resume():
    engine = FakeEngine()
    engine.release_terminal.clear()
    task = asyncio.create_task(run(engine))
    await asyncio.wait_for(engine.pause_seen.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.closed.is_set() and engine.paused and engine.events == []


@pytest.mark.parametrize("flat", [False, True])
def test_repeated_join_preserves_attempt_metadata_and_final_termination(flat):
    first, second, third = (
        completion([10]),
        completion([20]),
        completion([30], "B", "stop"),
    )
    first.sampling_mask = SamplingMask([[10]])
    third.stop_reason = 30
    if flat:
        for attempt in (first, second, third):
            logs = FlatLogprobs()
            logs.append(attempt.logprobs[0])
            attempt.logprobs = logs
    partial = join_attempts(
        [first, completion([]), second], logprobs_mode="raw_logprobs"
    )
    result = join_attempts(
        [*partial.attempts, third], logprobs_mode=partial.logprobs_mode
    )
    assert result.weight_versions == [
        WeightVersionSpan("A", 0, 2),
        WeightVersionSpan("B", 2, 3),
    ]
    assert result.sampled_token_logprobs == [-0.1, -0.2, -0.3]
    assert result.logprobs_mode == partial.logprobs_mode
    assert result.attempts[0] is first and first.sampling_mask.token_ids == [[10]]
    assert first.weight_versions == [WeightVersionSpan("A", 0, 1)]
    assert result.finish_reason == "stop" and result.stop_reason == 30


@pytest.mark.parametrize(
    "invalid", ["unfinished", "missing_spans", "gap", "missing_logprobs"]
)
def test_join_rejects_incomplete_attribution(invalid):
    attempt = completion([10, 20])
    if invalid == "unfinished":
        attempt.finish_reason = None
    elif invalid == "missing_spans":
        attempt.weight_versions = None
    elif invalid == "gap":
        attempt.weight_versions = [WeightVersionSpan("A", 1, 2)]
    else:
        attempt.logprobs = []
    with pytest.raises(ValueError):
        join_attempts([attempt], logprobs_mode="raw_logprobs")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"n": 2},
        {"presence_penalty": 1},
        {"stop": ["END"]},
        {"bad_words": ["bad"]},
        {
            "repetition_detection": RepetitionDetectionParams(
                max_pattern_size=3, min_count=2
            )
        },
        {"structured_outputs": StructuredOutputsParams(regex="[0-9]+")},
        {"max_tokens": None},
    ],
)
async def test_unsupported_continuation_rejected_before_generating(options):
    engine = FakeEngine()
    with pytest.raises(ValueError):
        await run(engine, SamplingParams(**{"max_tokens": 6, **options}))
    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raw_logits", "processed_logits"])
async def test_logits_modes_rejected_before_generation_or_join(mode):
    engine = FakeEngine()
    engine.model_config.logprobs_mode = mode
    with pytest.raises(ValueError, match="logits"):
        await run(engine)
    assert engine.calls == []
    with pytest.raises(ValueError, match="logits"):
        join_attempts([completion([10])], logprobs_mode=mode)
