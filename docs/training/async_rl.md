# Async Reinforcement Learning

## Overview

In a standard RL training loop, generation and training happen sequentially: the policy generates rollouts, then training runs on those rollouts, and the cycle repeats. During generation the training accelerators sit idle, and vice versa.

The **one-off pipelining** approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

However, this overlap introduces a complication: weights must be updated in the inference engine mid-flight, while requests may still be in progress.

## The Pause and Resume API

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods. These let the trainer coordinate a clean window for weight synchronization without losing in-flight work.

### pause_generation

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

The `mode` parameter controls how in-flight requests are handled:

| Mode | Behavior |
| ---- | -------- |
| `"abort"` | Abort all in-flight requests immediately and return partial results (default) |
| `"wait"` | Wait for all in-flight requests to finish before pausing |
| `"keep"` | Freeze requests in the queue; they resume when `resume_generation` is called |

The `clear_cache` parameter controls whether to clear the KV cache and prefix cache after pausing.

### resume_generation

```python
await engine.resume_generation()
```

Resumes the scheduler after a pause. Any requests frozen with `mode="keep"` will continue generating.

### HTTP Endpoints

With `VLLM_SERVER_DEV_MODE=1`, the vLLM HTTP server exposes the same functionality via:

- `POST /pause?mode=keep` - Pause generation
- `POST /resume` - Resume generation
- `POST /abort_requests` - Abort in-flight requests without pausing the scheduler (send `{}` to abort all, or `{"request_ids": [...]}`)
- `POST /update_weight_version` - Set the `weight_version` label without changing weights (send `{"new_version": "v8"}`)
- `GET /weight_info` - Return the latest committed `weight_version`

!!! note "Data Parallelism"
    When using data parallelism with vLLM's **internal load balancer** (i.e. `data_parallel_backend="ray"`), pause and resume are handled automatically across all DP ranks -- a single call is sufficient. When using an **external load balancer** (i.e. multiple independent vLLM instances behind a proxy), you must send pause and resume requests to **every** engine instance individually before and after the weight update, and set the weight version on every instance too.

## Typical Async RL Flow

### Abort and resubmit

An abort-first controller can stop requests at a weight-update boundary and save their partial rollouts:

1. Await `pause_generation(mode="abort", clear_cache=True)` on the relevant engines.
2. Collect each request's terminal output, including an empty final streaming chunk. The pause acknowledgment alone does not mean the client has consumed that output.
3. Save the partial tokens, their original sampled-token logprob values and `logprobs_mode`, and `weight_versions`.
4. Transfer the new weights and await `finish_weight_update(weight_version=...)` before resuming generation.
5. Resubmit each aborted request with the retained tokens appended to its prompt and the remaining token budget.
6. Append the new output to the logical rollout. Shift its spans by the number of previously retained output tokens, and preserve the original logprobs for those tokens.

For example, an aborted request returning `A [0, 30)` followed by a new request returning `B [0, 70)` becomes `A [0, 30), B [30, 100)` in the trainer's rollout. Prompt and tool-observation tokens need their own training masks; they are not generated-token version spans.

The [reference controller helper](../../examples/rl/abort_resume_weight_versions.py) demonstrates one cycle with a dedicated `AsyncLLM`, `n=1`, cumulative outputs, and ordinary sampling. The caller supplies sampling parameters and weight transfer between the helper's start/finish calls, and can use `save_partial` to persist the prefix before the update. It is not a complete training loop or a checkpoint of RNG, penalty, or structured-decoding state. The helper records the engine's `logprobs_mode`; raw logprobs must not be assumed to include temperature/top-p processing. If an update fails, the controller must recover before resuming.

### Keep in-flight requests

A loop that preserves requests in the engine instead looks like this:

1. Start generating rollouts from the current policy
2. Once trainer has new weights to update to, pause generation with `mode="keep"`
3. Sync the updated weights from the trainer to the inference engine (see [Weight Transfer](weight_transfer/README.md)) and set their weight version
4. Resume generation -- in-flight requests continue with the new weights
5. Repeat

The key insight is that requests paused with `mode="keep"` will produce tokens from the **old** weights before the pause and tokens from the **new** weights after resume. The `clear_cache` parameter controls whether the KV cache is invalidated during the pause. When `clear_cache=True`, previously cached key-value entries are discarded, so all tokens generated after resume will be computed entirely with the new weights. When `clear_cache=False`, existing KV cache entries are retained, meaning some tokens in context may still reflect the old weights (stale KV cache).

## Weight Versions in Outputs

To tell which tokens came from which weights, set a weight-version label while generation is paused, after the new weights are loaded: pass `weight_version` to `finish_weight_update`, or call `update_weight_version` (`POST /update_weight_version` over HTTP). Labels are strings you choose, such as the trainer step; vLLM doesn't order or check them.

The final output of every sequence then carries `weight_versions`, one span per label over its output-token indexes. A request paused with `mode="keep"` across one update gets two spans:

```python
completion = final_output.outputs[0]
for span in completion.weight_versions:
    # The tokens in [span.start, span.end) were sampled by weights `span.version`.
    print(span.version, completion.token_ids[span.start : span.end])
```

The offsets cover the whole output, so with streamed delta outputs, collect the tokens before slicing. `/inference/v1/generate` returns the same list on each choice; see [Tokens In <> Tokens Out API](../serving/online_serving/token_in_token_out.md#weight-versions).

OpenAI chat and completions requests can opt in with `return_weight_versions: true`. Each choice then carries `weight_versions` on the full response or its final streaming chunk, including aborts. Other responses omit the field. Set `return_token_ids: true` too when the controller needs exact generated-token coordinates. With echoed prompts, span indexes still follow the raw generated `token_ids`, not rendered text; echo-only prompt scoring can expose an internal sample in those IDs. Use `echo=false` for rollout collection.

The Python `LLM` and `AsyncLLM` `finish_weight_update(weight_version=...)` paths commit the label in the same EngineCore call that finishes the worker update. Worker failures and draft-only updates leave the target label unchanged. This is a per-engine ordering guarantee, not a transaction across independent replicas; the controller must keep failed updates paused. The initial label must be set before admitting rollouts, and version-to-checkpoint mapping remains the controller's responsibility.

## Example

The [async RLHF example](../../examples/rl/rlhf_async_new_apis.py) demonstrates this pattern with `vllm.AsyncLLMEngine`, NCCL weight transfer, and mid-flight pause/resume with validation.

For the implementation rationale, support boundaries, and follow-up plan, see the [weight provenance design reference](../design/async_rl_weight_provenance_reference.md).
