from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn
from vllm032 import test_fp4 as gate
from vllm.models.deepseek_v41.nvidia.tilelang_fp4 import attention as native
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.workspace import WorkspaceManager


def warmup():
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v41.nvidia.tilelang_fp4.cache_writer import writer_call
    from vllm.models.deepseek_v41.nvidia.tilelang_fp4.prepare import KERNELS

    registry = JitWarmupRegistry(None)
    call = writer_call("swa", 32, torch.int64)
    with registry.activate():
        KERNELS.register_warmup([call, call])
    with patch("vllm.distributed.is_global_first_rank", return_value=True):
        with patch.object(
            KERNELS, "compile_many", wraps=KERNELS.compile_many
        ) as compile_many:
            registry.warmup()
    assert len(compile_many.call_args.args[0]) == 1
    values = torch.randn((3, 512), device="cuda").bfloat16()
    cache = torch.zeros((1, 32 * 528), dtype=torch.uint8, device="cuda")
    native.write_cache(
        values,
        cache,
        torch.arange(3, device="cuda"),
        page_size=32,
        cache_kind="swa",
    )
    actual = cache.view(32, 528)[:3]
    torch.testing.assert_close(
        actual.cpu(), gate.reference(values, "swa"), rtol=0, atol=0
    )
    print(
        "real JIT registry compiles and launches byte-exact native kernels", flush=True
    )


def metadata():
    config = NS(scheduler_config=NS(max_num_batched_tokens=16))
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=68,
        dtype=torch.uint8,
        tokens_per_state=2,
        alignment=512,
    )
    builder = native.NativeIndexerMetadataBuilder(
        spec, ["cache"], config, torch.device("cuda")
    )
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    common = NS(
        num_actual_tokens=3,
        num_reqs=2,
        max_query_len=2,
        max_seq_len=260,
        query_start_loc=starts.cuda(),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([260, 258], dtype=torch.int32, device="cuda"),
        positions=torch.tensor([259, 256, 257], device="cuda"),
        slot_mapping=torch.tensor([9, 8, -1], device="cuda"),
        block_table_tensor=torch.tensor(
            [[2, 0], [1, 3]], dtype=torch.int32, device="cuda"
        ),
    )
    for operation in [
        lambda: builder.build(0, common),
        lambda: builder.build_for_cudagraph_capture(common),
    ]:
        result = operation()
        assert result.slot_mapping.tolist() == [1, -1, -1], result.slot_mapping
        assert (
            result.num_decodes,
            result.num_prefills,
            result.num_decode_tokens,
            result.num_prefill_tokens,
        ) == (1, 1, 1, 2)
    assert native.NativeIndexerBackend.get_supported_kernel_block_sizes() == [256]
    print("native metadata decode, prefill, replay PAD and capture passed", flush=True)


def padding():
    layer = native.NativeAttention.__new__(native.NativeAttention)
    nn.Module.__init__(layer)
    layer._native_ready = True
    layer._native_swa_width = 128
    layer._native_is_draft = False
    layer.compress_ratio = 1
    layer.is_kv_source = True
    layer.max_model_len = 1024
    layer._native_index_args = (1024, 256, False, False)
    layer._native_index = native.IndexerV41Plan(1024, 256)
    layer._native_mla = native.SparseMlaV41Plan(
        16, 128, 512, 256, swa_page_size=32, max_rows=16
    )
    layer._native_candidates = torch.empty((3, 16384), dtype=torch.int32, device="cuda")
    layer._native_counts = torch.empty(3, dtype=torch.int32, device="cuda")
    layer.topk_indices_buffer = torch.zeros((3, 512), dtype=torch.int32, device="cuda")
    index_cache = torch.zeros((4, 256, 68), dtype=torch.uint8, device="cuda")
    keys = torch.randn(1024, 128, device="cuda").bfloat16() * 0.2
    native.write_index_keys(
        keys, native.pages(index_cache), torch.arange(1024, device="cuda"), 256
    )
    layer.indexer = NS(k_cache=NS(prefix="index", kv_cache=index_cache))
    layer.swa_cache_layer = NS(
        prefix="swa",
        kv_cache=torch.zeros((3, 32, 528), dtype=torch.uint8, device="cuda"),
    )
    native.write_cache(
        torch.randn(96, 512, device="cuda").bfloat16() * 0.2,
        layer.swa_cache_layer.kv_cache,
        torch.arange(96, device="cuda"),
        page_size=32,
        cache_kind="swa",
    )
    layer.kv_cache = torch.zeros((4, 256, 288), dtype=torch.uint8, device="cuda")
    native.write_cache(
        torch.randn(1024, 512, device="cuda").bfloat16() * 0.2,
        layer.kv_cache,
        torch.arange(1024, device="cuda"),
        page_size=256,
        cache_kind="indexed",
    )
    layer.compressed_cache_prefix = "main"
    layer.attn_sink = nn.Parameter(torch.zeros(16, device="cuda"), requires_grad=False)
    layer.rotary_emb = NS(cos_sin_cache=torch.zeros((1024, 64), device="cuda"))
    table = torch.arange(4, dtype=torch.int32, device="cuda").expand(2, -1).contiguous()
    swa_indices = torch.full((3, 1, 128), -1, dtype=torch.int32, device="cuda")
    swa_indices[:, :, 0] = torch.tensor(
        [[0], [1], [2]], dtype=torch.int32, device="cuda"
    )
    swa = NS(
        num_decode_tokens=3,
        num_prefill_tokens=0,
        decode_swa_indices=swa_indices,
        decode_swa_lens=torch.ones(3, dtype=torch.int32, device="cuda"),
        is_valid_token=torch.tensor([True, False, True], device="cuda"),
        token_to_req_indices=torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda"),
    )
    context = NS(
        attn_metadata={
            "swa": swa,
            "index": NS(block_table=table),
            "main": NS(block_table=table),
        }
    )
    query = torch.randn(4, 16, 512, device="cuda").bfloat16() * 0.2
    iq = torch.randn(3 * 32, 128, device="cuda").bfloat16() * 0.2
    packed = torch.empty((3 * 32, 64), dtype=torch.uint8, device="cuda")
    scales = torch.empty((3 * 32, 4), dtype=torch.uint8, device="cuda")
    native.quantize_mxfp4(iq, packed, scales)
    weights = torch.full((3, 32), 1 / 64, dtype=torch.bfloat16, device="cuda")
    out = torch.full_like(query, 999)
    positions = torch.full((4,), 599, dtype=torch.int64, device="cuda")
    manager = WorkspaceManager(torch.device("cuda"))
    with (
        patch.object(native, "get_forward_context", return_value=context),
        patch.object(native, "current_workspace_manager", return_value=manager),
    ):
        layer._sparse_indexer_and_attn(
            torch.zeros((4, 1), device="cuda"),
            packed.reshape(3, 32, 64),
            scales.reshape(3, 32, 4),
            weights,
            query,
            None,
            positions,
            out,
        )
    assert bool((layer.topk_indices_buffer[1] == -1).all())
    assert bool((out[1] == 0).all())
    assert bool((out[3] == 999).all())
    assert bool(torch.isfinite(out[:3]).all())
    print(
        "mixed valid and poisoned PAD rows select no keys and produce zero; padded capture tail untouched",
        flush=True,
    )


def writer_padding():
    values = torch.randn(4, 512, device="cuda").bfloat16() * 0.2
    slots = torch.tensor([0, -1, 2], device="cuda")
    positions = torch.tensor([1, 3, 5, 7], device="cuda")
    cosine = torch.zeros((8, 64), device="cuda")
    cosine[:, :32] = 1
    rotary = NS(cos_sin_cache=cosine)
    swa = native.NativeAttention.__new__(native.NativeAttention)
    nn.Module.__init__(swa)
    swa.rotary_emb = rotary
    swa.swa_cache_layer = NS(
        block_size=32,
        kv_cache=torch.zeros((1, 32, 528), dtype=torch.uint8, device="cuda"),
    )
    swa.insert_swa_kv(values, positions, slots)
    expected = torch.zeros((32, 528), dtype=torch.uint8)
    records = gate.reference(values[:3], "swa")
    expected[0], expected[2] = records[0], records[2]
    torch.testing.assert_close(
        swa.swa_cache_layer.kv_cache[0].cpu(), expected, rtol=0, atol=0
    )
    cache = torch.zeros((1, 128, 288), dtype=torch.uint8, device="cuda")
    compressor = native.NativeCompressor.__new__(native.NativeCompressor)
    nn.Module.__init__(compressor)
    compressor.compress_ratio = 2
    compressor.k_cache_prefix = "main"
    compressor._static_forward_context = {"main": NS(kv_cache=cache)}
    context = NS(
        attn_metadata={"main": NS(slot_mapping=slots), "index": NS(slot_mapping=slots)}
    )
    with patch.object(native, "get_forward_context", return_value=context):
        compressor.insert_cache(values, positions, rotary)
    expected = torch.zeros((128, 288), dtype=torch.uint8)
    records = gate.reference(values[:3], "indexed")
    expected[0], expected[2] = records[0], records[2]
    torch.testing.assert_close(cache[0].cpu(), expected, rtol=0, atol=0)
    index = native.NativeIndexer.__new__(native.NativeIndexer)
    nn.Module.__init__(index)
    index.compress_ratio = 2
    index.k_cache = NS(
        prefix="index",
        kv_cache=torch.zeros((1, 128, 68), dtype=torch.uint8, device="cuda"),
    )
    index.wk = lambda x: (x[:, :128].contiguous(), None)
    index.k_norm = lambda x: x
    with patch.object(native, "get_forward_context", return_value=context):
        index._produce_k(values, positions, rotary)
    expected = torch.zeros((128 * 68,), dtype=torch.uint8)
    records = gate.reference(values[:3, :128], "index")
    for row in (0, 2):
        expected[row * 64 : (row + 1) * 64] = records[row, :64]
        expected[128 * 64 + row * 4 : 128 * 64 + (row + 1) * 4] = records[row, 64:]
    torch.testing.assert_close(
        index.k_cache.kv_cache.flatten().cpu(), expected, rtol=0, atol=0
    )
    print(
        "SWA, compressed and index cache writes exclude padded rows and preserve PAD slots",
        flush=True,
    )


if __name__ == "__main__":
    warmup()
    metadata()
    padding()
    writer_padding()
