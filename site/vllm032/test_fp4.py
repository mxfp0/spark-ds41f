import json
import torch

from vllm.models.deepseek_v41.nvidia.tilelang_fp4.cache_writer import write_cache
from vllm.models.deepseek_v41.nvidia.tilelang_fp4.indexer import (
    IndexerV41Plan,
    quantize_mxfp4,
    write_index_keys,
)
from vllm.models.deepseek_v41.nvidia.tilelang_fp4.sparse_mla import SparseMlaV41Plan

torch.manual_seed(1907)
LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def fp4_codes(values):
    distances = (values.abs().unsqueeze(-1) - LEVELS).abs()
    tied = distances == distances.min(-1, keepdim=True).values
    codes = torch.arange(8).expand_as(distances)
    even = tied & (codes % 2 == 0)
    chosen = torch.where(even.any(-1, keepdim=True), even, tied)
    result = torch.where(chosen, codes, 99).min(-1).values.to(torch.uint8)
    return result | (torch.signbit(values).to(torch.uint8) << 3)


def unpack_fp4(data):
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2)
    return LEVELS[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)


def reference(values, kind):
    values = values.float().cpu()
    group = 16 if kind == "indexed" else 32
    blocks = values.reshape(values.shape[0], -1, group)
    if kind == "indexed":
        scales = (
            (blocks.abs().amax(-1).clamp_min(6 * 2**-9) / 6)
            .clamp_max(448)
            .to(torch.float8_e4m3fn)
        )
        codes = fp4_codes(blocks / scales.float().unsqueeze(-1)).flatten(1)
        data = codes[:, ::2] | (codes[:, 1::2] << 4)
        return torch.cat((data, scales.view(torch.uint8)), -1)
    maximum = blocks.abs().amax(-1).clamp_min(1e-4 if kind == "swa" else 6 * 2**-126)
    normalized = maximum * (1 / (448 if kind == "swa" else 6))
    bits = normalized.contiguous().view(torch.int32).long()
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0)
    scales = torch.ldexp(torch.ones_like(maximum), exponent.int() - 127)
    normalized = blocks / scales.unsqueeze(-1)
    if kind == "swa":
        data = normalized.to(torch.float8_e4m3fn).view(torch.uint8).flatten(1)
    else:
        codes = fp4_codes(normalized).flatten(1)
        data = codes[:, ::2] | (codes[:, 1::2] << 4)
    return torch.cat((data, exponent.to(torch.uint8)), -1)


def dequant(records, kind):
    if kind == "indexed":
        data, scales = (
            records[:, :256],
            records[:, 256:].view(torch.float8_e4m3fn).float(),
        )
        return (unpack_fp4(data).reshape(-1, 32, 16) * scales.unsqueeze(-1)).reshape(
            -1, 512
        )
    if kind == "swa":
        data, scales = (
            records[:, :512].contiguous().view(torch.float8_e4m3fn).float(),
            records[:, 512:],
        )
        return (
            data.reshape(-1, 16, 32)
            * torch.ldexp(
                torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
            ).unsqueeze(-1)
        ).reshape(-1, 512)
    data, scales = records[:, :64], records[:, 64:]
    return (
        unpack_fp4(data).reshape(-1, 4, 32)
        * torch.ldexp(
            torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
        ).unsqueeze(-1)
    ).reshape(-1, 128)


def writers():
    for page in (32, 128, 256):
        for kind, width in [("swa", 528), ("indexed", 288)]:
            values = (torch.randn(8, 512) * 0.23).to(torch.bfloat16)
            values[0].zero_()
            values[1, :16] = torch.tensor(
                [
                    0,
                    -0.0,
                    0.25,
                    -0.25,
                    0.75,
                    -0.75,
                    1.25,
                    -1.25,
                    1.75,
                    -1.75,
                    2.5,
                    -2.5,
                    3.5,
                    -3.5,
                    5,
                    -5,
                ]
            )
            slots = torch.tensor(
                [-1, 0, 1, page - 1, page, page + 1, 2 * page - 1, 2 * page],
                device="cuda",
            )
            pool = torch.full(
                (3, page * width + 512), 205, dtype=torch.uint8, device="cuda"
            )
            write_cache(values.cuda(), pool, slots, page_size=page, cache_kind=kind)
            expected = torch.full(pool.shape, 205, dtype=torch.uint8)
            records = reference(values, kind)
            for row, slot in enumerate(slots.cpu().tolist()):
                if slot >= 0:
                    expected[
                        slot // page, slot % page * width : (slot % page + 1) * width
                    ] = records[row]
            torch.testing.assert_close(pool.cpu(), expected, rtol=0, atol=0)
            print("writer", kind, page, "exact bytes and padding passed", flush=True)


def attention():
    for page in (128, 256):
        x = (torch.randn(96, 512) * 0.3).to(torch.bfloat16)
        y = (torch.randn(page * 3, 512) * 0.3).to(torch.bfloat16)
        swa = torch.zeros((3, 32 * 528 + 512), dtype=torch.uint8, device="cuda")
        indexed = torch.zeros((3, page * 288 + 512), dtype=torch.uint8, device="cuda")
        write_cache(
            x.cuda(),
            swa,
            torch.arange(96, device="cuda"),
            page_size=32,
            cache_kind="swa",
        )
        write_cache(
            y.cuda(),
            indexed,
            torch.arange(page * 3, device="cuda"),
            page_size=page,
            cache_kind="indexed",
        )
        x = dequant(reference(x, "swa"), "swa")
        y = dequant(reference(y, "indexed"), "indexed")
        for compressed in (False, True):
            plan = SparseMlaV41Plan(
                16, 128, 512 if compressed else 0, page, swa_page_size=32, max_rows=256
            )
            for rows in (1, 6, 65):
                q = (torch.randn(rows, 16, 512) * 0.15).to(torch.bfloat16).cuda()
                sinks = torch.linspace(-1, 1, 16).cuda()
                slots = torch.full((rows, 128), -1, dtype=torch.int32, device="cuda")
                slots[:, :5] = torch.tensor([31, 32, 63, 64, 95], device="cuda")
                lengths = torch.full((rows,), 5, dtype=torch.int32, device="cuda")
                chosen = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
                chosen[:, :5] = torch.tensor(
                    [0, 3, page, page + 2, 2 * page], device="cuda"
                )
                tables = (
                    torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
                    .expand(rows, -1)
                    .contiguous()
                )
                values = x[slots[0, :5].cpu().long()]
                if compressed:
                    physical = torch.tensor([2 * page, 2 * page + 3, 0, 2, page])
                    values = torch.cat((values, y[physical]), 0)
                logits = q.cpu().float() @ values.T / 512**0.5
                logits = torch.cat(
                    (logits, sinks.cpu().expand(rows, 16).unsqueeze(-1)), -1
                )
                expected = (logits.softmax(-1)[..., : values.shape[0]] @ values).to(
                    torch.bfloat16
                )
                args = (
                    dict(
                        idx_cache=indexed,
                        idx_indices=chosen,
                        idx_lengths=lengths,
                        idx_page_table=tables,
                    )
                    if compressed
                    else {}
                )
                output = torch.empty_like(q)
                positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
                cosine = torch.zeros((1, 64), device="cuda")
                plan.run(
                    q, swa, slots, lengths, sinks, positions, cosine, output, **args
                )
                torch.testing.assert_close(
                    output.cpu(), expected, atol=0.002, rtol=0.02
                )
                if rows == 1:
                    scratch = [
                        torch.empty(shape, dtype=dtype, device="cuda")
                        for shape, dtype in plan.scratch_specs(rows)
                    ]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            plan.run(
                                q,
                                swa,
                                slots,
                                lengths,
                                sinks,
                                positions,
                                cosine,
                                output,
                                scratch=scratch,
                                **args,
                            )
                    torch.cuda.current_stream().wait_stream(stream)
                    q.mul_(0.5)
                    graph.replay()
                    logits = q.cpu().float() @ values.T / 512**0.5
                    logits = torch.cat(
                        (logits, sinks.cpu().expand(rows, 16).unsqueeze(-1)), -1
                    )
                    expected = (logits.softmax(-1)[..., : values.shape[0]] @ values).to(
                        torch.bfloat16
                    )
                    torch.testing.assert_close(
                        output.cpu(), expected, atol=0.002, rtol=0.02
                    )
                print(
                    "attention",
                    page,
                    compressed,
                    rows,
                    "CPU oracle and replay passed",
                    flush=True,
                )


def indexer():
    page, width, rows = 256, 1024, 2
    keys = (torch.randn(width, 128) * 0.3).bfloat16()
    queries = (torch.randn(rows, 32, 128) * 0.2).bfloat16()
    pool = torch.full((4, page * 68 + 512), 205, dtype=torch.uint8, device="cuda")
    write_index_keys(keys.cuda(), pool, torch.arange(width, device="cuda"), page)
    records = reference(keys, "index")
    expected = torch.full(pool.shape, 205, dtype=torch.uint8)
    for p in range(4):
        group = records[p * page : (p + 1) * page]
        expected[p, : page * 64] = group[:, :64].flatten()
        expected[p, page * 64 : page * 68] = group[:, 64:].flatten()
    torch.testing.assert_close(pool.cpu(), expected, rtol=0, atol=0)
    packed = torch.empty((rows * 32, 64), dtype=torch.uint8, device="cuda")
    scales = torch.empty((rows * 32, 4), dtype=torch.uint8, device="cuda")
    quantize_mxfp4(queries.cuda().reshape(-1, 128), packed, scales)
    qref = reference(queries.reshape(-1, 128), "index")
    torch.testing.assert_close(
        torch.cat((packed.cpu(), scales.cpu()), -1), qref, rtol=0, atol=0
    )
    k = dequant(records, "index")
    q = dequant(qref, "index").reshape(rows, 32, 128)
    weights = (torch.rand(rows, 32) / 64).bfloat16()
    contrib = (
        ((q @ k.T).bfloat16().float().clamp_min(0) * weights.float().unsqueeze(-1))
        .bfloat16()
        .float()
    )
    scores = torch.zeros(rows, width)
    for head in range(32):
        scores += contrib[:, head]
    scores = scores.bfloat16()
    plan = IndexerV41Plan(width, page)
    lengths = torch.tensor([width, 73], dtype=torch.int32, device="cuda")
    table = (
        torch.arange(4, dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
    )
    out, actual = plan.run(
        packed.reshape(rows, 32, 64),
        scales.reshape(rows, 32, 4),
        weights.cuda(),
        pool,
        table,
        lengths,
        torch.tensor([width], dtype=torch.int32, device="cuda"),
    )
    for row, n in enumerate(lengths.cpu().tolist()):
        torch.testing.assert_close(
            actual[row, :n].cpu(), scores[row, :n], rtol=0, atol=0
        )
        expected_indices = (
            torch.argsort(scores[row, :n], descending=True, stable=True)[:512]
            .sort()
            .values.int()
        )
        torch.testing.assert_close(
            out[row, : min(n, 512)].cpu(), expected_indices, rtol=0, atol=0
        )
        assert bool((out[row, min(n, 512) :] == -1).all())
    print(
        "indexer exact bytes, scores, ties, top-k and short padding passed", flush=True
    )


if __name__ == "__main__":
    writers()
    attention()
    indexer()
    print(json.dumps({"passed": True}), flush=True)
