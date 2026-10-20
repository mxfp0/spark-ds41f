import torch
from torch.utils._python_dispatch import TorchDispatchMode


class AnonymousCheckpointCopy(TorchDispatchMode):
    """Stage pageable TP shards through anonymous memory for GB10/R580."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.copy_.default:
            destination, source = args[:2]
            if (
                destination.is_cuda
                and source.device.type == "cpu"
                and not source.is_pinned()
            ):
                args = (destination, source.clone(), False)
        return func(*args, **kwargs)
