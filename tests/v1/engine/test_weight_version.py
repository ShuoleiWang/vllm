# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.rl.offline import RLOfflineMixin
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    DPLBAsyncMPClient,
    InprocClient,
    SyncMPClient,
)
from vllm.v1.engine.llm_engine import LLMEngine


@pytest.fixture
def core():
    core = object.__new__(EngineCore)
    core._weight_version = "old"
    core._weight_update_is_draft = False
    core.model_executor = Mock()
    core.model_executor.collective_rpc.return_value = ["worker-result"]
    return core


@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("version", [None, "new", ""])
@pytest.mark.parametrize("fails", [False, True])
def test_finish_commits_only_successful_target_updates(core, is_draft, version, fails):
    """Draft updates and failed finishes must not relabel target tokens."""
    start = "start_draft_weight_update" if is_draft else "start_weight_update"
    core.collective_rpc(start)

    def finish(method, *args):
        assert method == "finish_weight_update"
        assert core.get_weight_version() == "old"
        if fails:
            raise RuntimeError("worker finish failed")

    core.model_executor.collective_rpc.side_effect = finish
    if fails:
        with pytest.raises(RuntimeError, match="worker finish failed"):
            core.finish_weight_update(version)
        assert core.get_weight_version() == "old"
        # Retry without a new start must retain the original update target.
        core.model_executor.collective_rpc.side_effect = None

    core.finish_weight_update(version)
    expected = "old" if is_draft or version is None else version
    assert core.get_weight_version() == expected


@pytest.mark.parametrize("is_draft", [False, True])
def test_failed_start_preserves_current_update_target(core, is_draft):
    start = "start_draft_weight_update" if is_draft else "start_weight_update"
    other_start = "start_weight_update" if is_draft else "start_draft_weight_update"
    core.collective_rpc(start)
    core.model_executor.collective_rpc.side_effect = RuntimeError("already active")
    with pytest.raises(RuntimeError, match="already active"):
        core.collective_rpc(other_start)
    core.model_executor.collective_rpc.side_effect = None
    core.finish_weight_update("new")
    assert core.get_weight_version() == ("old" if is_draft else "new")


def test_raw_finish_preserves_worker_result_and_weight_label(core):
    core.collective_rpc("start_draft_weight_update")
    result = core.collective_rpc("finish_weight_update", timeout=12)
    assert result == ["worker-result"]
    assert core.get_weight_version() == "old"
    core.model_executor.collective_rpc.assert_called_with(
        "finish_weight_update", 12, (), None
    )
    core.collective_rpc("start_weight_update")
    core.finish_weight_update("new")
    assert core.get_weight_version() == "new"


@pytest.mark.parametrize("client_cls", [InprocClient, SyncMPClient])
@pytest.mark.parametrize("is_draft", [False, True])
def test_offline_finish_reaches_core_in_one_call(core, client_cls, is_draft):
    """Both sync clients must preserve the target across the public RL API."""
    client = object.__new__(client_cls)
    calls = []

    def dispatch(method, *args):
        calls.append(method)
        return getattr(core, method)(*args)

    if client_cls is InprocClient:
        client.engine_core = core
    else:
        client.call_utility = dispatch
    engine = object.__new__(LLMEngine)
    engine.engine_core = client
    llm = SimpleNamespace(llm_engine=engine)
    start = (
        RLOfflineMixin.start_draft_weight_update
        if is_draft
        else RLOfflineMixin.start_weight_update
    )
    start(llm)
    RLOfflineMixin.finish_weight_update(llm, "new")
    assert core.get_weight_version() == ("old" if is_draft else "new")
    if client_cls is SyncMPClient:
        assert calls == ["collective_rpc", "finish_weight_update"]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AsyncMPClient, DPLBAsyncMPClient])
@pytest.mark.parametrize("is_draft", [False, True])
async def test_async_finish_reaches_every_managed_core(core, client_cls, is_draft):
    """AsyncLLM uses one utility per core, including internal DP fanout."""
    client = object.__new__(client_cls)
    client.core_engine = b"rank-0"
    client.core_engines = [b"rank-0"]
    cores = {b"rank-0": core}
    if client_cls is DPLBAsyncMPClient:
        second = object.__new__(EngineCore)
        second._weight_version = "old"
        second._weight_update_is_draft = False
        second.model_executor = Mock()
        client.core_engines.append(b"rank-1")
        cores[b"rank-1"] = second

    async def dispatch(method, *args, engine):
        return getattr(cores[engine], method)(*args)

    client._call_utility_async = AsyncMock(side_effect=dispatch)
    llm = SimpleNamespace(engine_core=client)
    llm.collective_rpc = partial(AsyncLLM.collective_rpc, llm)
    start = (
        AsyncLLM.start_draft_weight_update if is_draft else AsyncLLM.start_weight_update
    )
    await start(llm)
    await AsyncLLM.finish_weight_update(llm, "new")
    expected = "old" if is_draft else "new"
    assert all(engine.get_weight_version() == expected for engine in cores.values())
    calls = client._call_utility_async.await_args_list
    assert [call.args[0] for call in calls].count("finish_weight_update") == len(cores)
    assert len(calls) == 2 * len(cores)
