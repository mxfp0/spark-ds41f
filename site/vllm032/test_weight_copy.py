def main():
    import gc
    import torch
    from vllm032.weight_copy import AnonymousCheckpointCopy

    source = torch.arange(1024 * 1024, dtype=torch.float32).reshape(-1, 1024)[:, ::2]
    pinned = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
    for options in [{}, {"non_blocking": True}]:
        destination = torch.empty(source.shape, dtype=source.dtype, device="cuda")
        with AnonymousCheckpointCopy():
            destination.copy_(source, **options)
            other = torch.empty_like(source)
            other.copy_(source)
            gpu = torch.empty_like(destination)
            gpu.copy_(destination)
        gc.collect()
        assert torch.equal(destination.cpu(), source)
        assert torch.equal(gpu.cpu(), source)
        assert torch.equal(other, source)
        torch.cuda.synchronize()
    assert torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) == pinned
    print(
        "Copy options, strided shard, pass-through and no retained pinned memory passed"
    )


if __name__ == "__main__":
    main()
