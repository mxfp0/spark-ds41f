import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

from .disk import from_local_table, restore_heads


class DiskEmbedding(ParallelEngramEmbedding):
    def __init__(self, num_embeddings, dim, head_sizes, max_tokens):
        nn.Module.__init__(self)
        if dim != 256:
            raise ValueError('Disk Engram requires 256-byte rows')
        self.cpu_offload = False
        self.dim = dim
        self.dp_size = 1
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = self.n_hash_cols // self.tp_size
        self.register_parameter('weight', None)
        self.register_parameter('weight_scale_inv', None)
        self.disk = from_local_table(num_embeddings, head_sizes,
                                     get_tensor_model_parallel_rank(), self.tp_size, max_tokens)

    @eager_break_during_capture
    def lookup(self, indices, out, background=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Disk Engram lookup cannot run inside an unbroken CUDA graph')
        if BreakableCUDAGraphCapture.is_active() or get_forward_context().attn_metadata is None:
            out.zero_()
            return
        rows = self.disk.lookup(indices.to(device='cpu', dtype=torch.int64).numpy())
        out.copy_(rows, non_blocking=False)

    def restore_head_order(self, rows):
        return restore_heads(rows, self.tp_size)
