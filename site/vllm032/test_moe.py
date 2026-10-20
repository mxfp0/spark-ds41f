# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused MXFP4 MoE checks based on official vLLM test_b12x.py."""

from dataclasses import replace
from types import SimpleNamespace
import importlib.metadata
import json

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_deepseek_v4_mxfp4_moe_backend
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.v1.worker.workspace import init_workspace_manager

def dq(packed, scales, dtype):
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                           -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                          device=packed.device, dtype=dtype)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    weight = values[codes].unflatten(-1, (-1, 32))
    factor = torch.exp2(scales.float() - 127).to(dtype)
    return (weight * factor.unsqueeze(-1)).flatten(-2)


dtype = torch.bfloat16
parallel = replace(FusedMoEParallelConfig.make_no_parallel(), tp_size=4)

def config(experts, topk, max_tokens=1024):
    return FusedMoEConfig(
        num_experts=experts, experts_per_token=topk, hidden_dim=5120,
        intermediate_size=2304, num_local_experts=experts,
        num_logical_experts=experts, activation=MoEActivation.SILU,
        device='cuda', routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=parallel, in_dtype=dtype,
        router_logits_dtype=torch.float32, moe_backend='b12x',
        max_num_tokens=max_tokens, swiglu_limit=10.0,
    )

def reference(x, w1, w2, weights, ids, clamp):
    output = torch.zeros_like(x, dtype=torch.float32)
    clamped = 0
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if not tokens.numel():
            continue
        projected = x[tokens] @ w1[expert].T
        gate, up = projected.chunk(2, dim=-1)
        clamped += int(((gate > 10) | (up.abs() > 10)).sum())
        if clamp:
            gate, up = gate.clamp(max=10), up.clamp(-10, 10)
        intermediate = F.silu(gate) * up
        result = intermediate @ w2[expert].T
        output.index_add_(0, tokens, result.float() * weights[tokens, slots, None])
    return output.to(dtype), clamped

@torch.inference_mode()
def main():
    assert importlib.metadata.version('b12x') == '1.5.1'
    init_workspace_manager(torch.device('cuda'))
    for name, experts, topk in [('target', 384, 6), ('draft', 128, 3)]:
        backend, cls = select_deepseek_v4_mxfp4_moe_backend(config(experts, topk))
        assert backend.value == 'B12X_MXFP4_MXFP8' and cls is B12xExperts
        print(json.dumps({'oracle': name, 'backend': backend.value, 'class': cls.__name__}), flush=True)

    torch.manual_seed(521)
    experts, hidden, intermediate = 8, 5120, 576
    w1, s1 = mxfp4_quantize(torch.randn(experts, intermediate * 2, hidden, device='cuda', dtype=dtype) / 15)
    w2, s2 = mxfp4_quantize(torch.randn(experts, hidden, intermediate, device='cuda', dtype=dtype) / 15)
    w1_ref = torch.stack([dq(w1[e], s1[e], dtype) for e in range(experts)])
    w2_ref = torch.stack([dq(w2[e], s2[e], dtype) for e in range(experts)])
    quant = FusedMoEQuantConfig.make(quant_dtype='mxfp8', weight_dtype='mxfp4',
                                   w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0)
    cfg = config(experts, 6, 32)
    implementation = B12xExperts(cfg, quant)
    layer = SimpleNamespace(activation=MoEActivation.SILU, apply_router_weight_on_input=False,
                            w13_weight=w1, w2_weight=w2, w13_weight_scale=s1, w2_weight_scale=s2)
    implementation.process_weights_after_loading(layer)
    implementation.warmup_launches(layer, token_counts=(1, 7, 32))
    kernel = FusedMoEKernel(maybe_make_prepare_finalize(moe=cfg, quant_config=quant,
                           allow_new_interface=True, use_monolithic=False), implementation)
    for tokens, topk, magnitude in [(1, 6, 0.1), (7, 3, 0.1), (32, 6, 0.1), (7, 6, 4.0)]:
        x = torch.randn(tokens, hidden, device='cuda', dtype=dtype) * magnitude
        score = torch.randn(tokens, experts, device='cuda', dtype=dtype)
        weights, ids, _ = fused_topk(x, score, topk, renormalize=False)
        expected, clamped = reference(x, w1_ref, w2_ref, weights, ids, True)
        actual = kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=weights,
                              topk_ids=ids, activation=MoEActivation.SILU,
                              global_num_experts=experts, expert_map=None,
                              apply_router_weight_on_input=False)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        error = actual.float() - expected.float()
        relative = float(error.norm() / expected.float().norm())
        cosine = float(F.cosine_similarity(actual.flatten().float(), expected.flatten().float(), dim=0))
        assert cosine > 0.99 and relative < 0.1, (tokens, topk, magnitude, cosine, relative)
        if magnitude < 1:
            torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)
        else:
            unclamped, _ = reference(x, w1_ref, w2_ref, weights, ids, False)
            assert clamped > 0 and (expected.float() - unclamped.float()).norm() > expected.float().norm()
        print(json.dumps({'tokens': tokens, 'topk': topk, 'input_scale': magnitude,
                          'cosine': cosine, 'relative_l2_error': relative,
                          'clamped_elements': clamped, 'max_abs_error': float(error.abs().max())}), flush=True)
    for tokens, topk in [(1, 6), (7, 3)]:
        def apply(x, weights, ids):
            return kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=weights,
                                topk_ids=ids, activation=MoEActivation.SILU,
                                global_num_experts=experts, expert_map=None,
                                apply_router_weight_on_input=False)
        x = torch.randn(tokens, hidden, device='cuda', dtype=dtype) * 0.1
        score = torch.randn(tokens, experts, device='cuda', dtype=dtype)
        weights, ids, _ = fused_topk(x, score, topk, renormalize=False)
        first = apply(x, weights, ids)
        snapshot = first.clone()
        alternate_x = torch.randn_like(x) * 0.1
        alternate_score = torch.randn_like(score)
        alternate_weights, alternate_ids, _ = fused_topk(alternate_x, alternate_score, topk, renormalize=False)
        second = apply(alternate_x, alternate_weights, alternate_ids)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
        assert first.data_ptr() != second.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                apply(x, weights, ids)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = apply(x, weights, ids)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, snapshot, atol=0, rtol=0)
        previous = captured.clone()
        for multiplier in [0.1, 4.0]:
            x.copy_(torch.randn_like(x) * multiplier)
            new_weights, new_ids, _ = fused_topk(x, torch.randn_like(score), topk, renormalize=False)
            weights.copy_(new_weights)
            ids.copy_(new_ids)
            expected, clamped = reference(x, w1_ref, w2_ref, weights, ids, True)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(captured).all()
            cosine = float(F.cosine_similarity(captured.flatten().float(), expected.flatten().float(), dim=0))
            relative = float((captured.float() - expected.float()).norm() / expected.float().norm())
            assert cosine > 0.99 and relative < 0.1, (tokens, topk, multiplier, cosine, relative)
            assert not torch.equal(captured, previous)
            previous.copy_(captured)
            print(json.dumps({'graph_tokens': tokens, 'topk': topk, 'changed_input_scale': multiplier,
                              'cosine': cosine, 'relative_l2_error': relative,
                              'clamped_elements': clamped, 'scratch_ownership': 'pass'}), flush=True)
    print('B12X_NUMERICAL_SMOKE_PASS', flush=True)

if __name__ == '__main__':
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))):
        main()
