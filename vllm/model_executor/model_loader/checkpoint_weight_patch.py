# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply dense or sparse updates expressed in checkpoint coordinates.

Calling ``model.load_weights`` lets each model keep control of checkpoint-name
mapping, packed parameters, and parallel slicing. Sparse updates support
floating-point loaders whose final write is one same-shaped ``copy_`` into each
non-overlapping runtime view.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_DEFAULT_PATCH_CHUNK_BYTES = 512 << 20

__all__ = [
    "CheckpointWeightPatch",
    "load_checkpoint_weight_patches",
]


@dataclass(frozen=True)
class CheckpointWeightPatch:
    """One dense replacement or sparse update in checkpoint coordinates.

    Attributes:
        name: Checkpoint weight name passed to the model loader.
        shape: Full checkpoint tensor shape.
        dtype: Checkpoint tensor dtype.
        values: Flattened replacement or patch values.
        indices: Flattened global positions in ``shape``. ``None`` means that
            ``values`` replaces the full tensor.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    values: torch.Tensor
    indices: torch.Tensor | None = None


class _SparsePatchCopyMode(TorchDispatchMode):
    """Validate where the model loader finally copies each checkpoint weight.

    Model loaders may rename a weight, slice it for TP, or place it in part of
    a packed parameter. Each supported copy merges the NaN-masked checkpoint
    source with the runtime destination before writing it. Each runtime view
    must receive exactly one same-shaped floating-point copy.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._destinations: list[torch.Tensor] = []
        self._runtime_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*model.parameters(), *model.buffers())
            if tensor.numel()
        }

    @staticmethod
    def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
        first = tensor.storage_offset()
        last = first
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            extent = (size - 1) * stride
            first += min(0, extent)
            last += max(0, extent)
        element_size = tensor.element_size()
        storage_start = tensor.untyped_storage().data_ptr()
        return (
            storage_start + first * element_size,
            storage_start + (last + 1) * element_size,
        )

    @classmethod
    def _views_may_overlap(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        if left.device != right.device:
            return False
        left_start, left_end = cls._storage_range(left)
        right_start, right_end = cls._storage_range(right)
        return left_start < right_end and right_start < left_end

    def _is_runtime_tensor(self, value) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.numel() > 0
            and value.untyped_storage().data_ptr() in self._runtime_storages
        )

    @staticmethod
    def _iter_tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _SparsePatchCopyMode._iter_tensors(item)

    def _written_runtime_tensors(self, func, args, kwargs):
        for index, argument in enumerate(func._schema.arguments):
            alias_info = argument.alias_info
            if alias_info is None or not alias_info.is_write:
                continue
            if index < len(args):
                value = args[index]
            elif argument.name in kwargs:
                value = kwargs[argument.name]
            else:
                continue
            yield from (
                tensor
                for tensor in self._iter_tensors(value)
                if self._is_runtime_tensor(tensor)
            )

    def _record_destination(
        self,
        destination: torch.Tensor,
    ) -> None:
        for previous in self._destinations:
            if self._views_may_overlap(destination, previous):
                raise NotImplementedError(
                    "sparse checkpoint patches cannot write overlapping runtime views"
                )
        self._destinations.append(destination)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        runtime_writes = list(self._written_runtime_tensors(func, args, kwargs))
        if func is torch.ops.aten.copy_.default and runtime_writes:
            dst, src = args[:2]
            if (
                isinstance(src, torch.Tensor)
                and dst.is_floating_point()
                and src.is_floating_point()
                and dst.shape == src.shape
            ):
                cast = src.to(dtype=dst.dtype, device=dst.device)
                self._record_destination(dst)
                merged = torch.where(torch.isnan(cast), dst, cast)
                args = (dst, merged, *args[2:])
            else:
                raise NotImplementedError(
                    "sparse checkpoint patches require a same-shaped floating-point "
                    "copy into each runtime destination"
                )
        elif runtime_writes:
            raise NotImplementedError(
                "sparse checkpoint patches only support copy_ writes to "
                f"runtime weights; got {func}"
            )
        return func(*args, **kwargs)

    def validate(self, *, allow_no_write: bool) -> None:
        if not self._destinations and not allow_no_write:
            raise NotImplementedError(
                "the model loader reported a sparse checkpoint weight without "
                "copying it into a runtime destination"
            )


def _validate_patch(patch: CheckpointWeightPatch) -> int:
    if not patch.name:
        raise ValueError("Checkpoint patch names must be non-empty")
    if not isinstance(patch.dtype, torch.dtype):
        raise TypeError(f"{patch.name}: dtype must be a torch.dtype")
    if any(dim < 0 for dim in patch.shape):
        raise ValueError(f"{patch.name}: shape dimensions must be non-negative")
    if patch.values.ndim != 1:
        raise ValueError(f"{patch.name}: patch values must be flattened")

    numel = math.prod(patch.shape)
    if patch.indices is None:
        if patch.values.numel() != numel:
            raise ValueError(
                f"{patch.name}: dense patch has {patch.values.numel()} values "
                f"for a {numel}-element tensor"
            )
        return numel

    if not patch.dtype.is_floating_point:
        raise TypeError(
            f"{patch.name}: sparse checkpoint patches require a floating dtype "
            "because NaN is the unchanged-value sentinel"
        )
    if patch.indices.ndim != 1:
        raise ValueError(f"{patch.name}: sparse patch indices must be flattened")
    if patch.indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{patch.name}: sparse patch indices must be int32 or int64")
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(
            f"{patch.name}: sparse indices and values must have matching lengths"
        )

    return numel


def _load_chunk(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
    *,
    sparse: bool,
) -> set[str]:
    if not weights:
        return set()
    if sparse:
        mode = _SparsePatchCopyMode(model)
        with mode:
            loaded = model.load_weights(weights)
        allow_no_write = loaded is not None and not loaded
        mode.validate(allow_no_write=allow_no_write)
    else:
        loaded = model.load_weights(weights)
    return set() if loaded is None else set(loaded)


@torch.no_grad()
def load_checkpoint_weight_patches(
    model: torch.nn.Module,
    patches: Iterable[CheckpointWeightPatch],
    *,
    max_chunk_bytes: int = _DEFAULT_PATCH_CHUNK_BYTES,
    validate_unique_indices: bool = True,
) -> set[str]:
    """Load ordered patches through the model's checkpoint loader.

    Dense and sparse patches cannot be mixed. Duplicate names keep their input
    order so callers can split one large parameter across several patches.
    Each sparse patch materializes a full checkpoint-shaped NaN staging tensor.
    ``max_chunk_bytes`` controls how many such tensors share one loader call;
    one tensor may exceed it.

    The caller wraps dense updates in vLLM's layerwise reload lifecycle. Sparse
    updates modify initialized runtime tensors and must not run that lifecycle.
    Metadata is checked before loading. Loader failures propagate without
    restoring destinations, so callers must stop using and restart a worker
    after a failed update.

    Args:
        model: Model whose native ``load_weights`` method applies the patches.
        patches: Dense replacements or sparse updates in checkpoint coordinates.
        max_chunk_bytes: Target size for each checkpoint staging chunk.
        validate_unique_indices: Whether to reject duplicate sparse indices.
            Disable only for a trusted producer that already guarantees unique
            positions.

    Returns:
        Names reported as loaded by the model loader.

    Raises:
        ValueError: If patch metadata is invalid or dense and sparse patches
            are mixed.
        TypeError: If a patch uses an unsupported dtype or tensor layout.
        NotImplementedError: If a sparse model loader does not finish with one
            supported copy into each runtime view.
    """

    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")

    patch_list = list(patches)
    sizes = [_validate_patch(patch) for patch in patch_list]
    sparse_modes = {patch.indices is not None for patch in patch_list}
    if len(sparse_modes) > 1:
        raise ValueError("Dense and sparse checkpoint patches cannot be mixed")
    sparse = sparse_modes == {True}

    validation_flags: dict[torch.device, list[torch.Tensor]] = {}
    if sparse:
        for patch, numel in zip(patch_list, sizes, strict=True):
            assert patch.indices is not None
            flags = validation_flags.setdefault(patch.values.device, [])
            flags.append(torch.isnan(patch.values).any())
            indices = patch.indices.to(device=patch.values.device)
            if indices.numel():
                flags.append(torch.logical_or(indices < 0, indices >= numel).any())
                if validate_unique_indices:
                    sorted_indices = torch.sort(indices).values
                    flags.append((sorted_indices[1:] == sorted_indices[:-1]).any())
        for flags in validation_flags.values():
            if bool(torch.stack(flags).any().item()):
                raise ValueError(
                    "sparse checkpoint patches contain NaN values, out-of-range "
                    "indices, or duplicate indices"
                )

    loaded_names: set[str] = set()
    chunk: list[tuple[str, torch.Tensor]] = []
    chunk_names: set[str] = set()
    chunk_bytes = 0

    for patch, numel in zip(patch_list, sizes, strict=True):
        tensor_bytes = numel * torch.empty((), dtype=patch.dtype).element_size()
        if chunk and (
            chunk_bytes + tensor_bytes > max_chunk_bytes or patch.name in chunk_names
        ):
            loaded_names.update(_load_chunk(model, chunk, sparse=sparse))
            chunk = []
            chunk_names = set()
            chunk_bytes = 0

        if patch.indices is None:
            tensor = patch.values.to(dtype=patch.dtype).view(patch.shape)
        else:
            flat = torch.full(
                (numel,),
                float("nan"),
                dtype=patch.dtype,
                device=patch.values.device,
            )
            if patch.values.numel():
                flat.index_copy_(
                    0,
                    patch.indices.to(device=flat.device, dtype=torch.long),
                    patch.values.to(dtype=patch.dtype, device=flat.device),
                )
            tensor = flat.view(patch.shape)

        chunk.append((patch.name, tensor))
        chunk_names.add(patch.name)
        chunk_bytes += tensor_bytes

    loaded_names.update(_load_chunk(model, chunk, sparse=sparse))
    return loaded_names
