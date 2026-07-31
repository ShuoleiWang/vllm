# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    default_weight_loader,
)

pytestmark = pytest.mark.cpu_test


class _CheckpointPatchModel(torch.nn.Module):
    """Small loader that mimics a TP rank owning half of a checkpoint tensor."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))
        self.load_calls: list[list[str]] = []

    def load_weights(self, weights):
        weights = list(weights)
        self.load_calls.append([name for name, _ in weights])
        for name, loaded_weight in weights:
            assert name == "global.weight"
            self.weight.data.copy_(loaded_weight.narrow(0, 2, 2))
        return {name for name, _ in weights}


def _sparse_patch(
    *,
    index: int,
    value: float,
    shape: tuple[int, ...] = (4,),
) -> CheckpointWeightPatch:
    return CheckpointWeightPatch(
        name="global.weight",
        shape=shape,
        dtype=torch.float32,
        indices=torch.tensor([index], dtype=torch.int32),
        values=torch.tensor([value]),
    )


def test_sparse_patch_preserves_values_after_tp_narrow():
    model = _CheckpointPatchModel()
    patch = CheckpointWeightPatch(
        name="global.weight",
        shape=(4,),
        dtype=torch.float32,
        indices=torch.tensor([0, 3], dtype=torch.int32),
        values=torch.tensor([100.0, 30.0]),
    )

    loaded = load_checkpoint_weight_patches(model, [patch])

    assert torch.equal(model.weight, torch.tensor([0.0, 30.0]))
    assert loaded == {"global.weight"}


def test_dense_patch_uses_checkpoint_to_runtime_narrow():
    model = _CheckpointPatchModel()
    patch = CheckpointWeightPatch(
        name="global.weight",
        shape=(4,),
        dtype=torch.float32,
        values=torch.tensor([10.0, 11.0, 12.0, 13.0]),
    )

    load_checkpoint_weight_patches(model, [patch])

    assert torch.equal(model.weight, torch.tensor([12.0, 13.0]))


class _PackedCheckpointPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.packed_weight = torch.nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def load_weights(self, weights):
        loaded_names = set()
        for name, loaded_weight in weights:
            if name == "q_proj.weight":
                destination = self.packed_weight.data[:2]
            elif name == "k_proj.weight":
                destination = self.packed_weight.data[2:]
            else:
                raise AssertionError(name)
            destination.copy_(loaded_weight)
            loaded_names.add(name)
        return loaded_names


def test_sparse_patch_uses_packed_runtime_destinations():
    model = _PackedCheckpointPatchModel()
    patches = [
        CheckpointWeightPatch(
            name="q_proj.weight",
            shape=(2,),
            dtype=torch.float32,
            indices=torch.tensor([1], dtype=torch.int32),
            values=torch.tensor([10.0]),
        ),
        CheckpointWeightPatch(
            name="k_proj.weight",
            shape=(2,),
            dtype=torch.float32,
            indices=torch.tensor([0], dtype=torch.int32),
            values=torch.tensor([20.0]),
        ),
    ]

    load_checkpoint_weight_patches(model, patches)

    assert torch.equal(model.packed_weight, torch.tensor([0.0, 10.0, 20.0, 3.0]))


def test_duplicate_names_keep_input_order():
    model = _CheckpointPatchModel()
    patches = [
        _sparse_patch(index=2, value=20.0),
        _sparse_patch(index=3, value=30.0),
    ]

    load_checkpoint_weight_patches(model, patches)

    assert torch.equal(model.weight, torch.tensor([20.0, 30.0]))
    assert model.load_calls == [["global.weight"], ["global.weight"]]


class _ComposedCheckpointPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        raw = torch.tensor([0.0, 1.0])
        self.weight = torch.nn.Parameter(-torch.exp(raw))
        self.weight.weight_loader = composed_weight_loader(
            default_weight_loader,
            lambda value: -torch.exp(value),
        )

    def load_weights(self, weights):
        for name, loaded_weight in weights:
            assert name == "global.weight"
            self.weight.weight_loader(self.weight, loaded_weight)
        return {"global.weight"}


def test_composed_loader_is_rejected_and_rolled_back():
    model = _ComposedCheckpointPatchModel()
    original = model.weight.detach().clone()
    patch = _sparse_patch(index=0, value=2.0, shape=(2,))

    with pytest.raises(NotImplementedError, match="overlapping runtime views"):
        load_checkpoint_weight_patches(model, [patch])

    assert torch.equal(model.weight, original)


class _PostCopyMutationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def load_weights(self, weights):
        for _, loaded_weight in weights:
            self.weight.data.copy_(loaded_weight)
            self.weight.data.fill_(7.0)
        return {"global.weight"}


def test_post_copy_mutation_is_rejected_and_rolled_back():
    model = _PostCopyMutationModel()
    original = model.weight.detach().clone()

    with pytest.raises(NotImplementedError, match="only support copy_"):
        load_checkpoint_weight_patches(
            model,
            [_sparse_patch(index=0, value=20.0, shape=(2,))],
        )

    assert torch.equal(model.weight, original)


class _UnsupportedCopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def load_weights(self, weights):
        for _, loaded_weight in weights:
            self.weight.data.copy_(loaded_weight[:1])
        return {"global.weight"}


def test_unsupported_copy_is_rejected_and_rolled_back():
    model = _UnsupportedCopyModel()
    original = model.weight.detach().clone()
    patch = _sparse_patch(index=0, value=20.0, shape=(2,))

    with pytest.raises(NotImplementedError, match="same-shaped"):
        load_checkpoint_weight_patches(model, [patch])

    assert torch.equal(model.weight, original)


class _OverlappingCheckpointPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def load_weights(self, weights):
        for name, loaded_weight in weights:
            if name == "left.weight":
                destination = self.weight.data[:3]
            elif name == "right.weight":
                destination = self.weight.data[2:]
            else:
                raise AssertionError(name)
            destination.copy_(loaded_weight)
        return {name for name, _ in weights}


def test_overlapping_runtime_views_are_rejected_and_rolled_back():
    model = _OverlappingCheckpointPatchModel()
    original = model.weight.detach().clone()
    patches = [
        CheckpointWeightPatch(
            name="left.weight",
            shape=(3,),
            dtype=torch.float32,
            indices=torch.tensor([0], dtype=torch.int32),
            values=torch.tensor([10.0]),
        ),
        CheckpointWeightPatch(
            name="right.weight",
            shape=(2,),
            dtype=torch.float32,
            indices=torch.tensor([1], dtype=torch.int32),
            values=torch.tensor([30.0]),
        ),
    ]

    with pytest.raises(NotImplementedError, match="overlapping runtime views"):
        load_checkpoint_weight_patches(model, patches)

    assert torch.equal(model.weight, original)


class _IgnoredCheckpointPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def load_weights(self, weights):
        return {name for name, _ in weights}


def test_reported_weight_without_runtime_copy_is_rejected():
    model = _IgnoredCheckpointPatchModel()

    with pytest.raises(NotImplementedError, match="without copying"):
        load_checkpoint_weight_patches(
            model,
            [_sparse_patch(index=0, value=20.0, shape=(2,))],
        )


class _FailingCheckpointPatchModel(_CheckpointPatchModel):
    def load_weights(self, weights):
        super().load_weights(weights)
        raise RuntimeError("loader failed")


def test_loader_error_restores_destinations():
    model = _FailingCheckpointPatchModel()
    original = model.weight.detach().clone()

    with pytest.raises(RuntimeError, match="loader failed"):
        load_checkpoint_weight_patches(
            model,
            [_sparse_patch(index=3, value=30.0)],
        )

    assert torch.equal(model.weight, original)


def test_nan_values_are_rejected_before_loading():
    model = _CheckpointPatchModel()
    patch = _sparse_patch(index=2, value=float("nan"))

    with pytest.raises(ValueError, match="contain NaN"):
        load_checkpoint_weight_patches(model, [patch])

    assert model.load_calls == []


def test_duplicate_indices_are_rejected_before_loading():
    model = _CheckpointPatchModel()
    patch = CheckpointWeightPatch(
        name="global.weight",
        shape=(4,),
        dtype=torch.float32,
        indices=torch.tensor([2, 2], dtype=torch.int32),
        values=torch.tensor([20.0, 21.0]),
    )

    with pytest.raises(ValueError, match="duplicate indices"):
        load_checkpoint_weight_patches(model, [patch])

    assert model.load_calls == []


def test_all_metadata_is_validated_before_loading():
    model = _CheckpointPatchModel()
    patches = [
        _sparse_patch(index=2, value=20.0),
        _sparse_patch(index=4, value=40.0),
    ]

    with pytest.raises(ValueError, match="out-of-range"):
        load_checkpoint_weight_patches(model, patches)

    assert torch.equal(model.weight, torch.tensor([0.0, 1.0]))
    assert model.load_calls == []


def test_masked_copy_is_scoped_to_model_loading():
    model = _CheckpointPatchModel()
    load_checkpoint_weight_patches(
        model,
        [_sparse_patch(index=2, value=20.0)],
    )

    model.weight.data.copy_(torch.full_like(model.weight, float("nan")))
    assert torch.isnan(model.weight).all()


def test_failure_only_rolls_back_the_current_loader_call():
    model = _CheckpointPatchModel()
    patches = [
        _sparse_patch(index=2, value=20.0),
        _sparse_patch(index=3, value=30.0),
    ]
    original_load_weights = model.load_weights

    def fail_second_call(weights):
        if len(model.load_calls) == 1:
            original_load_weights(weights)
            raise RuntimeError("second chunk failed")
        return original_load_weights(weights)

    model.load_weights = fail_second_call

    with pytest.raises(RuntimeError, match="second chunk failed"):
        load_checkpoint_weight_patches(
            model,
            patches,
            max_chunk_bytes=1,
        )

    assert torch.equal(model.weight, torch.tensor([20.0, 1.0]))
