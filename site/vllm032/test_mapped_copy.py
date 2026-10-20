import json
import mmap
from pathlib import Path
import subprocess
import sys


def child():
    import torch
    from safetensors import safe_open
    from vllm032.weight_copy import AnonymousCheckpointCopy

    root = Path("/models")
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    reserve = torch.empty(72 * 1024**3, dtype=torch.uint8, device="cuda")
    mappings = []
    for name in sorted(set(index.values())):
        with (root / name).open("rb") as file:
            mappings.append(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY))
    with safe_open(str(root / index["embed.weight"]), framework="pt") as file:
        source = file.get_tensor("embed.weight")
        source = source.narrow(0, 0, source.shape[0] // 4)
        destination = torch.empty(source.shape, dtype=source.dtype, device="cuda")
        pinned = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
        with AnonymousCheckpointCopy():
            destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(destination.cpu(), source)
        assert (
            torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) == pinned
        )
        assert reserve.numel() == 72 * 1024**3 and len(mappings) == 48
    print(
        "Mapped checkpoint shard copy passed with bounded staging and no pinned allocation"
    )


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        subprocess.run([sys.executable, __file__, "--child"], check=True, timeout=90)
