# Async RL weight provenance: design reference and handoff

## Purpose and status

This branch is a design and implementation reference for future coding agents. It is built on [PR #59582](https://github.com/vllm-project/vllm/pull/59582), at `006ebac240f34ddcd36383832b28c25840d2ea80`. It is not a standalone upstream PR, a complete RL framework, or a claim that every part of [RFC #54443](https://github.com/vllm-project/vllm/issues/54443) is implemented. Keep the published PR separate from this follow-up branch.

The goal is to give asynchronous post-training systems reliable token provenance with small additions to existing vLLM output and weight-update paths. Prioritize abort/update/resubmit workflows. Treat keep, cache provenance, and more elaborate deployment combinations as separate extensions whose contracts need their own validation.

The implementation is deliberately split into engine mechanisms, API delivery, and a controller example. The controller belongs outside the engine: vLLM should not acquire trainer-specific freshness thresholds, checkpoint retention policies, or a distributed rollout scheduler.

## The training problem

A trainer can publish new checkpoints while agents are still producing long rollouts. A logical rollout may contain tokens from several checkpoints, including when every individual generation request used only one checkpoint. The training system needs to know which policy produced each sampled token so it can apply its chosen staleness filtering, token masks, or probability correction.

Reflection's [Beam announcement](https://reflection.ai/blog/introducing-beam), published October 5, 2026, describes per-token version tags and long rollouts spanning multiple checkpoints. At the time this reference was written, the announcement still said the full technical report would follow later that month. It does not disclose an abort/keep implementation or a complete correction algorithm. Use it as motivation for provenance, not evidence that this branch reproduces Beam's infrastructure or learning algorithm.

Three concepts must remain distinct:

- **Generation attempt:** one engine request, with its own output-token coordinates and terminal reason.
- **Logical rollout:** the trainer's trajectory, possibly assembled from several attempts and tool interactions.
- **Checkpoint label:** an opaque, controller-assigned string identifying the target weights. It is not itself a probability, a numerical staleness measure, or a certificate that all replicas hold identical tensors.

A version tag does not replace the sampled-token logprob values recorded during generation. Preserve those values together with the engine's `logprobs_mode`. The default `raw_logprobs` is computed before sampling transformations such as temperature and top-p; it must not automatically be interpreted as the final sampling distribution. `raw_logits` and `processed_logits` are not logprob modes. Recomputing an old prefix under new weights also produces values from a different checkpoint and must not overwrite the old records.

A trainer must choose and validate the probability convention required by its algorithm, including the sampling backend and transforms. This reference records the engine's logprob convention; it does not derive an importance-sampling formula or assert that every backend's processed values are interchangeable.

## The token contract

`CompletionOutput.weight_versions` contains half-open ranges over generated tokens, excluding the prompt:

```json
[
  {"version": "run-a/step-40", "start": 0, "end": 30},
  {"version": "run-a/step-41", "start": 30, "end": 100}
]
```

The spans are ordered, contiguous, and cover the retained output tokens. Adjacent equal labels can merge; A to B to A remains three spans. This is losslessly equivalent to an expanded per-token version array, with storage proportional to version transitions instead of repeated labels per token.

Each output sequence has its own coordinates. Zero generated tokens produce `[]`; missing provenance is `None`. Do not treat missing metadata as an empty output or infer it from the engine's later current label. The initial `"default"` label means the controller has not supplied checkpoint identity. Set an explicit label after loading the initial checkpoint and before admitting rollouts. Do not reuse a label for different weights within one training lineage.

Version spans use generated-token coordinates. Prompt tokens, tool observations, trainer loss/action masks, and routed-expert rows may use other coordinate systems. In particular, `sampling_mask` describes sampling support and is not the trainer's loss mask. Keep any required coordinate conversion explicit.

## Why abort comes first

An abort boundary ends the current attempt. Resuming scheduling does not resurrect that request. A controller may discard the partial attempt, train on it, or resubmit its retained prefix as part of a new prompt.

```text
Attempt 1: A produces 30 tokens, then aborts -> A [0,30)
Attempt 2: prompt includes that prefix; B produces 70 tokens -> B [0,70)
Logical rollout: A [0,30), B [30,100)
```

This already requires token-level provenance at the logical-rollout boundary. Keep is not a prerequisite for that requirement.

The reference update sequence is:

1. Pause the relevant engines with `mode="abort"` and an appropriate cache policy.
2. Continue consuming each attempt until its actual terminal output arrives. A pause acknowledgment is not a client-consumption acknowledgment.
3. Persist the retained tokens, original behavior logprobs, spans, and any replay data needed by the training algorithm.
4. Transfer the new weights and finish the update with the new checkpoint label.
5. Resume only after the relevant replicas have completed successfully.
6. Resubmit unfinished work with a fresh request ID, retained prefix, and remaining token budget. Shift the new attempt's spans when joining it into the logical rollout.

`/abort_requests` alone is a cancellation operation, not the pause/quiescence barrier for updating weights. A terminal abort chunk can contain no new tokens while carrying all final version spans. Clients must consume that metadata instead of filtering the chunk solely on text or token content.

## Implementation map and responsibilities

| Component | Responsibility in this branch |
| --- | --- |
| `EngineCoreOutputs`, `OutputProcessor`, `CompletionOutput` from #59582 | Carry the engine label, record version transitions, and build terminal spans. |
| `EngineCore.finish_weight_update` | Finish worker updates and then commit the target label within one engine utility call. |
| Sync and async engine clients | Route the existing public finish operation to that utility. Internal DP uses the existing broadcast mechanism. |
| OpenAI chat/completions | Deliver spans per choice when requested; preserve opt-out response shape. |
| `RayVLLMWeightSyncClient` | Send an optional label with the finish call rather than as a second remote update. |
| `abort_resume_weight_versions.py` | Demonstrate terminal collection, partial-result persistence, parameter preservation, and cross-attempt joining. |
| Training controller | Own global labels, admission/update coordination, logical rollout IDs, deduplication, persistence, recovery, and training policy. |

The principal source entry points are:

- [EngineCore](../../vllm/v1/engine/core.py), [engine clients](../../vllm/v1/engine/core_client.py), [AsyncLLM](../../vllm/v1/engine/async_llm.py), and [offline RL API](../../vllm/entrypoints/rl/offline.py).
- [Chat protocol](../../vllm/entrypoints/openai/chat_completion/protocol.py) and [serving](../../vllm/entrypoints/openai/chat_completion/serving.py).
- [Completion protocol](../../vllm/entrypoints/openai/completion/protocol.py) and [serving](../../vllm/entrypoints/openai/completion/serving.py).
- [Controller example](../../examples/rl/abort_resume_weight_versions.py) and [user-facing flow](../training/async_rl.md).

No new scheduler policy, GPU kernel, KV allocation scheme, or per-request model snapshot is introduced. The existing output metadata path remains the basis of attribution.

## Weight-update ordering and draft weights

The prototype previously awaited worker finish and then issued a separate label update. The refined path performs both actions in one EngineCore utility call: all worker finishes must return successfully before the label changes. Sync `LLM`, async `AsyncLLM`, and the Ray adapter reach this operation through their existing public APIs.

A small EngineCore flag records successful named `start_weight_update` and `start_draft_weight_update` RPCs. It changes only after the corresponding worker call succeeds. This reuses the established start path and worker-side session checks rather than introducing another update-session state machine. A successful draft-only finish does not change the target label, even if a label was supplied. Failed starts and finishes preserve the previous label and target state.

The ordinary named raw `collective_rpc("finish_weight_update")` keeps its worker return value and does not acquire implicit label changes. Manual `update_weight_version` remains an explicit caller override. Arbitrary callable RPCs that privately start update sessions cannot be classified by name; they are outside the automatic target/draft tracking contract.

This ordering guarantee is local to an EngineCore. It is not a fleet-wide transaction and does not roll back partially applied weights. A failed or timed-out update can leave physical state requiring recovery even though the label did not advance. Keep affected engines paused, recover or reload them, and confirm readiness before routing more work. There must be one coordinating update owner per engine; this branch does not provide controller election or update leases.

## OpenAI API contract

Use both flags when a training client needs exact generated-token coordinates:

```json
{
  "model": "policy",
  "prompt": [1, 2, 3],
  "max_tokens": 128,
  "stream": true,
  "return_token_ids": true,
  "return_weight_versions": true
}
```

The chat endpoint accepts the same opt-in metadata flag with `messages` instead of `prompt`.

- Opt-out responses omit `weight_versions`, including ordinary model serialization. Adding a field whose default serializes as `null` would change every existing response unnecessarily.
- Full responses include the field per choice. Streams include it on each choice's terminal chunk, including zero-token aborts; an optional trailing usage-only chunk does not repeat choice metadata.
- `n > 1` keeps independent spans for each choice. Text/tool parsing and rewritten finish reasons do not change the raw generated-token coordinates.
- Prompt echo does not shift spans. Coordinates follow the existing raw generation `token_ids`, not rendered text. An echo-only request with `max_tokens=0` can still expose an internal prompt-scoring sample in that raw field; its provenance is retained too, so the spans and token IDs agree. This branch does not rewrite the existing echo token/usage behavior. Use `echo=false` when collecting training rollouts.
- Beam search reconstructs completion objects without preserving this provenance. The combination of beam search and `return_weight_versions=true` is explicitly rejected rather than silently pretending to provide attribution. Beam search without the flag keeps its existing behavior.

Two existing empty-output filters now allow terminal events through. This is necessary for an abort before the first token to deliver its finish reason and empty span list. It does not add intermediate metadata to normal streams.

## Controller example: useful but intentionally bounded

The example accepts and copies caller `SamplingParams`, preserving ordinary settings such as temperature, top-p, top-k, seed, and token stop IDs. It requests cumulative outputs and sampled-token logprobs for reliable collection. The remaining generation budget and `min_tokens` are adjusted using the actual retained prefix length, which may exceed the requested abort threshold.

```python
from examples.rl.abort_resume_weight_versions import abort_and_resume_rollout
from vllm import SamplingParams

# The initial checkpoint is already loaded; no rollouts have been admitted yet.
await engine.update_weight_version("run-a/step-40")
params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=0.95)
rollout = await abort_and_resume_rollout(
    engine,
    prompt_token_ids,
    params,
    transfer_target_tensors,
    "run-a/step-41",
    abort_after=128,
    save_partial=persist_partial,
)
```

`transfer_target_tensors` and `persist_partial` are controller-provided awaitables. The helper owns start/finish; the transfer callback must only load the target tensors. Do not directly pass an existing helper that already performs an entire start/update/finish cycle, such as a trainer's `send_weights`, without adapting ownership. A controller that owns the whole update cycle can instead reuse `join_attempts` and the public engine APIs.

`Rollout` contains merged token IDs, `sampled_token_logprobs`, their `logprobs_mode`, merged version spans, and the terminal per-attempt `CompletionOutput` objects. Those attempts preserve optional sampling masks and routed experts in their original coordinates. Repeated cycles can reuse `join_attempts` on prior and new attempts with an explicit, common `logprobs_mode`; do not mix records from different conventions. The helper reads the engine's mode and rejects logits modes before generation. No trainer-specific rollout class is added to vLLM core.

The orchestration helper assumes a dedicated engine and `n=1`. Do not run one copy per request concurrently on a shared engine: pause acts on all requests. A real controller should pause once, collect all affected requests, and coordinate one update. The OpenAI output interface itself supports multiple choices.

The example rejects penalties, string stops, bad-word constraints, repetition detection, structured decoding, and prompt-scoring coordinates whose state cannot be restored simply by moving generated tokens into a new prompt. A seed starts a fresh request's RNG stream; exact equivalence to uninterrupted sampling is not promised. Custom stateful processors need an explicit continuation contract. Tool observations, rewards, loss masks, and duplicate-attempt handling remain controller data.

`save_partial` runs after terminal metadata validation and before the update starts. Save or update failures do not automatically resume the engine. Persisting only the successful function return would lose access to the prefix when an update fails. The callback should store a durable snapshot with controller-owned rollout/attempt identity.

## Deliberately deferred work

These items should not be inferred from the presence of the version field:

- **Initial-label configuration:** the engine still starts with `"default"`; initialize it explicitly before admission. A startup argument can be added independently if callers need it.
- **Queued in-process keep results:** #59582 stamps labels when outputs are processed. An in-process asynchronous batch can remain queued across a keep update and receive the later label. This known edge case is not fixed here; it needs a focused scheduling-time/batch snapshot change if that execution mode is required.
- **KV provenance:** a token tagged B may attend to KV computed under A when caches are retained. Prefix cache salting alone does not refresh a kept request's active KV. External cache coherence and P/D version skew need separate guarantees.
- **R3/P-D coverage:** [RFC #55584](https://github.com/vllm-project/vllm/issues/55584) addresses matching route records to the KV actually used. Its initial fixed-weight/DP=1 scope does not establish all MTP, DBO, layout, or context-parallel combinations.
- **Adapter identity:** target checkpoint labels do not version independently hot-swapped LoRA adapters. A served model name is not an immutable adapter revision; an adapter-specific provenance contract remains separate.
- **Other transports:** Rust response exposure, gRPC, and other response APIs require explicit propagation work. Decoding a wire field alone does not expose it to their callers.
- **Training algorithms and recovery:** version labels do not implement importance sampling, freshness thresholds, numerical consistency, checkpoint storage, or recovery across replica restarts.

## Validation and how to extend the reference

The following permanent tests are CPU-runnable and avoid model downloads:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/v1/engine/test_weight_version.py \
  tests/entrypoints/unit_tests/test_weight_versions.py \
  tests/entrypoints/rl/test_abort_resume_weight_versions.py
```

They exercise real protocol, serving, client-routing, and controller code with mocked execution boundaries. Coverage includes successful/failed target and draft updates; sync/async/DP utility routing; opt-in omission; terminal choice metadata; zero/partial aborts; echo/beam boundaries; parameter preservation; terminal-before-update ordering; saved partial results; cancellation; budgets; and version/logprob alignment.

Local validation for this reference passed 81 focused CPU test cases in the three files above, plus 20 existing output/API cases (101 total). Repository pre-commit checks, including mypy and test collection checks, passed. The broader `tests/distributed/test_weight_transfer.py` module could not be collected in this local environment because Ray was unavailable; the focused routing tests use mocked transport boundaries.

The tests are collected by the existing Engine, Entrypoints Unit, and RL test jobs. The RL job also depends on the example so changes to the helper trigger its tests. The existing Ray-client assertions describe the new one-call finish behavior; their larger module requires the Ray dependency.

Local CPU tests and pre-commit checks are not GPU or distributed weight-transfer validation. Before using this design in a training run, validate a real async engine and an actual checkpoint update, then test multiple abort cycles, slow consumers, request admission during pause, rank/update failure, retry/idempotence, and multi-replica rollout reconstruction. Compare model tokens and sampled logprobs with the intended behavior policy; do not validate provenance using label changes alone.

For a subsequent coding agent:

1. Re-read the current branch and relevant RFC revisions; do not treat this document's base commit as current upstream.
2. Select one bounded contract to improve. Keep engine mechanisms independent of a specific trainer's policy.
3. Extend the existing focused tests and report whether evidence is CPU, GPU, or real distributed execution.
4. Keep unsupported combinations explicit. Do not silently relabel unknown data or resume after an uncertain update.
5. Preserve this branch's reference-only publication intent unless the user separately requests an upstream PR.
