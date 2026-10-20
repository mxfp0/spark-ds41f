"""Kernel backend policy: which library runs the DS4.1 model compute kernels.

TileLang is the promoted backend (r6): it selects the TileLang compiler,
DeepSeek's TileKernels and the DS4.1 TileLang kernels in the vLLM patch stack,
with sparknet's one-shot collectives. The promoted configurations name it
explicitly. B12X remains the alternative; a configuration without
``kernel_backend`` means B12X, as every configuration recorded before r6 does.
The upstream backend uses official vLLM attention and linear kernels with
its optional B12X MoE package.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath


# The backend of a configuration that does not name one (every record before r6).
DEFAULT = "b12x"
# Serve arguments and the environment value each backend requires.
BACKENDS = {
    "upstream-fp4": {
        "--attention-backend": "TILELANG_DSV41",
        "--linear-backend": "flashinfer_b12x",
        "--moe-backend": "b12x",
    },
    "upstream": {
        "--attention-backend": "FLASHINFER_MLA_SPARSE_DSV41",
        "--linear-backend": "flashinfer_b12x",
        "--moe-backend": "b12x",
    },
    "b12x": {
        "--attention-backend": "B12X",
        "--linear-backend": "b12x",
        "--moe-backend": "b12x",
    },
    "tilelang": {
        "--attention-backend": "TILELANG",
        "--linear-backend": "tilelang",
        "--moe-backend": "tilelang",
    },
}
ENVIRONMENT = "VLLM_DS41_KERNEL_BACKEND"
CACHE_ENVIRONMENT = "TILELANG_CACHE_DIR"
# The TileLang backend needs these optional build sources, this capability in
# the vLLM source manifest record, and an image built from them.
SOURCES = ("tilelang", "tile_kernels")
CAPABILITY = "tilelang-kernels"


def backend(cluster: dict) -> str:
    return cluster.get("kernel_backend", DEFAULT)


def label(name: str) -> str:
    return f"local.spark3.{name}.tree"


def argument(cluster: dict, flag: str) -> str | None:
    args = cluster.get("serve_args", [])
    if flag not in args:
        return None
    i = args.index(flag) + 1
    return args[i] if i < len(args) else None


def problems(cluster: dict) -> list[str]:
    """Serve arguments and environment that contradict the selected backend."""
    selected = backend(cluster)
    if selected not in BACKENDS:
        return [f"unknown kernel_backend {selected!r}; choose {' or '.join(BACKENDS)}"]
    errors = []
    for flag, wanted in BACKENDS[selected].items():
        value = argument(cluster, flag)
        if value != wanted:
            errors.append(f"kernel_backend {selected} requires {flag} {wanted}, got {value}")
    try:
        spec = json.loads(argument(cluster, "--speculative-config") or "{}")
    except ValueError:
        spec = None
    if isinstance(spec, dict) and spec:
        wanted = BACKENDS[selected]["--attention-backend"]
        if spec.get("attention_backend") != wanted:
            errors.append(
                f"kernel_backend {selected} requires speculative attention_backend {wanted}, "
                f"got {spec.get('attention_backend')}"
            )
    env = cluster.get("environment", {})
    # vLLM defaults to B12X when the variable is absent, so only TileLang
    # requires it; a value that is present must name the selected backend.
    value = env.get(ENVIRONMENT)
    if selected in ("upstream", "upstream-fp4"):
        if value is not None:
            errors.append("upstream kernels do not use VLLM_DS41_KERNEL_BACKEND")
    elif (selected != DEFAULT or value is not None) and value != selected:
        errors.append(f"kernel_backend {selected} requires {ENVIRONMENT}={selected}")
    if selected == "tilelang":
        cache = PurePosixPath(str(env.get(CACHE_ENVIRONMENT, "")))
        if not cache.is_absolute() or not cache.is_relative_to("/cache") or cache == PurePosixPath("/cache"):
            errors.append(f"kernel_backend tilelang requires {CACHE_ENVIRONMENT} under /cache/")
    return errors


def source_problems(cluster: dict, upstreams: dict, manifest: dict) -> list[str]:
    """Required sources, capabilities and image trees for each backend."""
    selected = backend(cluster)
    official_image = selected in ("upstream", "upstream-fp4") or upstreams.get("sources", {}).get("vllm", {}).get("runtime_from_base_image") is True
    required = ("vllm", "vllm_base_image") if official_image else ("vllm", "b12x", "nccl")
    errors = [f"kernel_backend {selected} requires source {name}" for name in required
              if name not in upstreams.get("sources", {})]
    if official_image:
        tree = manifest.get("vllm", {}).get("expected_tree")
        labels = cluster.get("container", {}).get("expected_labels", {})
        if not tree or labels.get(label("vllm")) != tree:
            errors.append("upstream image tree must match its vLLM source manifest")
        if "disk-engram-local" not in manifest.get("vllm", {}).get("capabilities", []):
            errors.append("upstream image requires the disk-engram-local source capability")
        base = upstreams.get("sources", {}).get("vllm_base_image", {})
        if not str(base.get("recorded_base_digest", "")).startswith("sha256:"):
            errors.append("upstream base image must be pinned by digest")
        if not manifest.get("deployment_artifacts"):
            errors.append("upstream image must record its adapter source hashes")
        if selected in ("upstream-fp4", "tilelang") and "native-fp4-cache" not in manifest.get("vllm", {}).get("capabilities", []):
            errors.append(f"{selected} requires the native-fp4-cache source capability")
    if selected != "tilelang":
        return errors
    for name in SOURCES:
        if name not in upstreams.get("sources", {}):
            errors.append(f"kernel_backend tilelang requires source {name} in its upstreams lock")
    vllm = manifest.get("vllm", {})
    if CAPABILITY not in vllm.get("capabilities", []):
        errors.append(f"kernel_backend tilelang requires a vLLM source manifest with capability {CAPABILITY}")
    labels = cluster.get("container", {}).get("expected_labels", {})
    for name in ("vllm", *SOURCES):
        tree = manifest.get(name, {}).get("expected_tree")
        if tree is None or labels.get(label(name)) != tree:
            errors.append(
                f"kernel_backend tilelang requires container.expected_labels {label(name)} "
                "to match its source manifest tree"
            )
    return errors
