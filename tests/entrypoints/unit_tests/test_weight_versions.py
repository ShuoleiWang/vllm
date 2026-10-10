# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenAI weight-version delivery without a model, tokenizer, or engine process."""

import json
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.exceptions import VLLMValidationError
from vllm.outputs import CompletionOutput, RequestOutput, WeightVersionSpan

pytestmark = pytest.mark.skip_global_cleanup
SPAN = WeightVersionSpan("v7", 0, 1)
SPAN_JSON = [{"version": "v7", "start": 0, "end": 1}]


def _request(chat, **kwargs):
    if chat:
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hi"}], **kwargs
        )
    return CompletionRequest(prompt="Hi", **kwargs)


def _output(index=0, tokens=(), reason=None, spans=None):
    return CompletionOutput(
        index=index,
        text="x" * len(tokens),
        token_ids=list(tokens),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=reason,
        weight_versions=spans,
    )


def _result(*outputs):
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=list(outputs),
        finished=all(output.finished() for output in outputs),
    )


async def _serve(chat, results, parser=None, **kwargs):
    request = _request(chat, model="m", **kwargs)
    cls = OpenAIServingChat if chat else OpenAIServingCompletion
    serving: Any = cls.__new__(cls)
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_per_request_metrics = False
    serving.enable_log_outputs = False
    serving.system_fingerprint = None
    serving.response_role = "assistant"
    serving.enable_auto_tools = False
    serving._include_reasoning_tokens_details = False
    serving.parser_cls = Mock(tool_parser_cls=None) if parser else None
    serving._make_parser = Mock(return_value=parser)
    metadata = RequestResponseMetadata(request_id="r")

    async def source(indexed=False):
        for result in results:
            yield (0, result) if indexed else result

    if request.stream:
        if chat:
            chunks = serving.chat_completion_stream_generator(
                request,
                source(),
                "r",
                "m",
                [],
                Mock(),
                metadata,
            )
        else:
            chunks = serving.completion_stream_generator(
                request,
                [{"prompt": "Hi"}],
                source(indexed=True),
                "r",
                0,
                "m",
                1,
                None,
                metadata,
            )
        lines = [line async for line in chunks]
        assert lines[-1] == "data: [DONE]\n\n"
        return [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    if chat:
        response = await serving.chat_completion_full_generator(
            request,
            source(),
            "r",
            "m",
            [],
            None,
            metadata,
            parser=parser,
        )
    else:
        response = serving.request_output_to_completion_response(
            results,
            request,
            "r",
            0,
            "m",
            None,
            metadata,
        )
    return [json.loads(response.model_dump_json())]


@pytest.mark.parametrize(
    "choice",
    [
        CompletionResponseChoice(index=0, text=""),
        CompletionResponseStreamChoice(index=0, text=""),
        ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant")),
        ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage()),
    ],
)
def test_weight_versions_serialization_omits_unknown_but_preserves_empty(choice):
    for spans, expected in [(None, None), ([], []), ([SPAN], SPAN_JSON)]:
        choice = choice.model_copy(update={"weight_versions": spans})
        for options in (
            {},
            {"exclude_none": True},
            {"exclude_unset": True},
            {"exclude_defaults": True},
        ):
            for data in (
                choice.model_dump(**options),
                json.loads(choice.model_dump_json(**options)),
            ):
                if expected is None:
                    assert "weight_versions" not in data
                else:
                    assert data["weight_versions"] == expected


@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
def test_beam_search_rejects_weight_versions_only_when_requested(chat):
    assert not _request(chat, use_beam_search=True).return_weight_versions
    with pytest.raises(VLLMValidationError, match="not supported with beam search"):
        _request(chat, use_beam_search=True, return_weight_versions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("return_token_ids", [False, True])
async def test_full_weight_versions_are_per_choice_and_independent_of_token_ids(
    chat,
    opt_in,
    return_token_ids,
):
    second = [SPAN, WeightVersionSpan("v8", 1, 2)]
    (response,) = await _serve(
        chat,
        [
            _result(
                _output(0, reason="abort", spans=[]),
                _output(1, [10, 20], "stop", second),
            )
        ],
        n=2,
        return_weight_versions=opt_in,
        return_token_ids=return_token_ids,
    )
    if opt_in:
        assert response["choices"][0]["weight_versions"] == []
        assert response["choices"][1]["weight_versions"] == [
            *SPAN_JSON,
            {"version": "v8", "start": 1, "end": 2},
        ]
    else:
        assert all("weight_versions" not in choice for choice in response["choices"])


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("include_usage", [False, True])
async def test_stream_abort_emits_final_spans_for_each_choice(
    chat, opt_in, include_usage
):
    chunks = await _serve(
        chat,
        [
            _result(_output(0), _output(1)),  # Empty prefill is not termination.
            _result(_output(0, [10])),
            _result(_output(1, reason="abort", spans=[])),
            _result(_output(0, reason="abort", spans=[SPAN])),
        ],
        n=2,
        stream=True,
        return_weight_versions=opt_in,
        stream_options={"include_usage": include_usage},
    )
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    terminal = [choice for choice in choices if choice.get("finish_reason")]
    assert [choice["index"] for choice in terminal] == [1, 0]
    assert all(choice["finish_reason"] == "abort" for choice in terminal)
    for choice in choices:
        if opt_in and choice.get("finish_reason"):
            assert choice["weight_versions"] == (
                [] if choice["index"] == 1 else SPAN_JSON
            )
        else:
            assert "weight_versions" not in choice
    if include_usage:
        assert chunks[-1]["choices"] == []
        assert "weight_versions" not in chunks[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 2])
@pytest.mark.parametrize("return_token_ids", [False, True])
async def test_completion_echo_spans_follow_raw_generated_token_ids(
    stream,
    max_tokens,
    return_token_ids,
):
    # echo-only samples a token for prompt scoring; raw token_ids still include it.
    chunks = await _serve(
        False,
        [_result(_output(tokens=[10], reason="length", spans=[SPAN]))],
        stream=stream,
        echo=True,
        max_tokens=max_tokens,
        return_token_ids=return_token_ids,
        return_weight_versions=True,
    )
    choice = chunks[-1]["choices"][0]
    prompt_text = "" if return_token_ids else "Hi"
    assert choice["text"] == prompt_text + ("" if max_tokens == 0 else "x")
    assert choice["weight_versions"] == SPAN_JSON
    if return_token_ids:
        assert choice["prompt_token_ids"] == [1, 2]
        assert choice["token_ids"] == [10]
        assert choice["weight_versions"][0]["end"] == len(choice["token_ids"])


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["abort", "stop"])
async def test_chat_parser_without_text_still_emits_terminal_weight_versions(reason):
    parser = Mock()
    parser.parse_delta.return_value = None
    chunks = await _serve(
        True,
        [_result(_output(tokens=[10])), _result(_output(reason=reason, spans=[SPAN]))],
        stream=True,
        return_weight_versions=True,
        parser=parser,
        include_reasoning=False,
    )
    terminal = chunks[-1]["choices"][0]
    assert terminal["finish_reason"] == reason
    assert terminal["weight_versions"] == SPAN_JSON


@pytest.mark.asyncio
async def test_chat_tool_call_finish_keeps_weight_versions():
    parser = Mock()
    parser.parse_delta.side_effect = [
        DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(name="f", arguments="{}"),
                )
            ]
        ),
        None,
    ]
    chunks = await _serve(
        True,
        [_result(_output(tokens=[10])), _result(_output(reason="stop", spans=[SPAN]))],
        stream=True,
        return_weight_versions=True,
        parser=parser,
        tool_choice="required",
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    terminal = chunks[-1]["choices"][0]
    assert terminal["finish_reason"] == "tool_calls"
    assert terminal["weight_versions"] == SPAN_JSON
