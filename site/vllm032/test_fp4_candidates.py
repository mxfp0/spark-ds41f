import torch
from vllm032 import test_fp4 as gate
from vllm.models.deepseek_v41.nvidia.tilelang_fp4.indexer import (
    IndexerV41Plan,
    quantize_mxfp4,
    write_index_keys,
)


def main():
    torch.manual_seed(5678)
    width, page = 32768, 256
    keys = (torch.randn(width, 128) * 0.2).bfloat16()
    queries = (torch.randn(32, 128) * 0.2).bfloat16()
    weights = (torch.rand(1, 32) / 64).bfloat16()
    pool = torch.zeros(
        (width // page, page * 68 + 512), dtype=torch.uint8, device="cuda"
    )
    write_index_keys(keys.cuda(), pool, torch.arange(width, device="cuda"), page)
    packed = torch.empty((32, 64), dtype=torch.uint8, device="cuda")
    scales = torch.empty((32, 4), dtype=torch.uint8, device="cuda")
    quantize_mxfp4(queries.cuda(), packed, scales)
    q = gate.dequant(gate.reference(queries, "index"), "index")
    k = gate.dequant(gate.reference(keys, "index"), "index")
    contrib = (
        ((q @ k.T).bfloat16().float().clamp_min(0) * weights[0].float().unsqueeze(-1))
        .bfloat16()
        .float()
    )
    scores = torch.zeros(width)
    for h in range(32):
        scores += contrib[h]
    scores = scores.bfloat16()
    table = torch.arange(width // page, dtype=torch.int32, device="cuda").unsqueeze(0)
    lengths = torch.tensor([width - 3], dtype=torch.int32, device="cuda")
    active = torch.tensor([width], dtype=torch.int32, device="cuda")
    candidates = torch.full((1, 16384), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(1, dtype=torch.int32, device="cuda")
    source = IndexerV41Plan(width, page, candidate_blocks=True)
    chosen, actual = source.run(
        packed.unsqueeze(0),
        scales.unsqueeze(0),
        weights.cuda(),
        pool,
        table,
        lengths,
        active,
        candidate_out=candidates,
        candidate_out_lengths=counts,
    )
    torch.testing.assert_close(
        actual[0, : width - 3].cpu(), scores[: width - 3], rtol=0, atol=0
    )
    masked = scores.clone()
    masked[width - 3 :] = -torch.inf
    blocks = masked.reshape(-1, 8).amax(-1)
    blocks[-1] = torch.inf
    selected_blocks = (
        torch.argsort(blocks, descending=True, stable=True)[:2048].sort().values
    )
    expected = (selected_blocks[:, None] * 8 + torch.arange(8)).flatten()
    expected = expected[expected < width - 3].int()
    torch.testing.assert_close(
        candidates[0, : expected.numel()].cpu(), expected, rtol=0, atol=0
    )
    assert counts.item() == expected.numel()
    assert bool((candidates[0, expected.numel() :] == -1).all())
    reindex = IndexerV41Plan(16384, page, candidates=True)
    out, _ = reindex.run(
        packed.unsqueeze(0),
        scales.unsqueeze(0),
        weights.cuda(),
        pool,
        table,
        lengths,
        active,
        candidates,
        counts,
    )
    best = torch.argsort(scores[expected.long()], descending=True, stable=True)[:512]
    selected = expected[best].sort().values
    torch.testing.assert_close(out[0].cpu(), selected, rtol=0, atol=0)
    print(
        "source block selection, partial tail, hierarchical merges and reindex exact CPU oracle passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
