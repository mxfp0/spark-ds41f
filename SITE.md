# Four-Spark deployment

Active recipe: `/home/spark/ai/spark-ds41f` on spark1-spark4.
Use `bin/spark --cluster-config config/sparks.json doctor --live`.

## Model and client

Official `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, native MXFP4 experts.
The image is `deepseek1017-vllm:v032-native-tp4-64k-r8`, ID
`sha256:ab1e19bd6a82a84a7d123292631dd22dc78f7c70671555af8c2d692e78d9310b`.
It uses official vLLM 0.32.0, B12X 1.5.1, FlashInfer 0.7.1, TileLang 0.1.15,
TileKernels 2.0.0 and Sparknet 0.3.0. Patched source tree:
`acca1b7746612f485b54899446a44c05eaadedd3`.

Clients use llama-swap on port 8080, model `DeepSeek-V4.1-Flash-1017-vLLM`.
The direct backend is spark1:8000, model `deepseek-v4.1-flash`.
GLM remains available through the same router.

## Capacity

TP4/EP1, context 1048576, 16 streams, 10.5 GiB KV per rank, DSpark 5,
8 images per request. Logical KV blocks contain 256 tokens. The live engine
reports 8,580,566 shared logical KV tokens, exactly the qualified r6x capacity
at its original budget. Streams share that pool.

Native `TILELANG` retains 528-byte MXFP8 SWA, 288-byte NVFP4 compressed KV
and 68-byte MXFP4 indexer records. The interim 8-bit backend used 584-byte
compressed KV and 132-byte indexer records. Full native allocation restores
capacity at the original 10.5 GiB budget.
Startup 5 GiB and steady 3 GiB memory guards remain unchanged.

The current 1M context setting is not a new near 1M qualification. Sustained C16,
concurrent long contexts and full output-cap behavior remain unqualified.
SparkDash C32 queues 32 clients against the unchanged 16-slot engine limit.

## Restored runtime

The target and draft use native TileLang dense, MoE, mHC, RoPE, attention
and cache kernels. Routed experts retain lossless scale codes, FP32 shared
expert folding and decode L2 prefetch. The vocabulary head keeps exact packed
BF16 weights and FP32 logits. Checkpoint quantization is unchanged; optional
NVFP4 draft and Markov heads remain disabled.

Engram retains early host hashes, asynchronous local O_DIRECT reads,
GPU dequantization and capturable gather. Prefill keeps TP sequence
parallelism and CED compaction. Eligible collectives use Sparknet one-shot
ring4; other messages use the qualified paired-domain NCCL path.

Adaptive DSpark keeps fixed row cost 0.035, five draft slots, sharded main
and vocabulary projections, separate context graphs, padding-aware routing
and the final dead-row ownership clamp. Draft query padding is bounded by
actual decode capacity; context buffers retain full prefill capacity.

The native direct loader uses B12X 1.5.1 selected TP byte ranges and preserves
Engram descriptors and copy-completion fences. The original allocator,
CUDA connection limit 32, cache flags, scheduler threshold 8096 and no-reserve
policy are restored. Official vLLM 0.32 runner and allocator interfaces remain.

The site enables `SPARK3_GREEDY_MIN_TOKENS=1`. This optional extension applies
vLLM's exact minimum-token stop mask to fresh rank-local logits before
sharded greedy verification. It defaults off outside this profile. Logprobs,
grammars, custom samplers/processors and other unsupported processing retain
the full sampler. Exactness tests cover finite logits, boundaries and padding;
no general NaN-equivalence claim is made for the inherited greedy shortcut.

## Verification

Fresh replay of the three patches on official vLLM commit
`5df63a7203cdb9328d824a27b682e77c7b9928be` reproduces the image source tree.
Published patch mail uses normalized author metadata; the replayed source tree is
unchanged. Qualified image, source-commit and library identities remain historical.

All four ranks passed 123 runtime-file hashes, package versions, mapped NCCL
identity, local O_DIRECT Engram and cgroup OOM checks. Image IDs match.
Runtime NFS READ counts stayed unchanged on all three clients throughout the final benchmarks. Final live doctor reports zero errors
and 12 existing read-only inspection warnings.

The initial native kernel gate passed 118 GPU cases with one expected failure.
The final focused suite passed 68 cases on GPU; CPU execution passed 47 cases
with 21 GPU cases skipped. The direct-loader GPU fixture verified exact TP row,
column, scalar, widening, E8M0 and vocabulary-padding copies and copy fences.

Live arithmetic, tools and continuation, vision, reasoning, cached history,
C4, a 2400-pixel benchmark screenshot and router streamed/nonstreamed metrics
passed. Forced usage remains present even when the client disables it.
The actual 400-token SparkDash prompt returned identical content with the
fast path and full-logprob fallback, also identical to retained r6x and R7.
The fallback returned all 400 logprob rows. An untimed stack sample positively
observed the minimum-token mask and sharded verification. The default-locking py-spy profiler stalled a worker; it was stopped,
both requests completed correctly, and
all ranks passed recovery checks. Its timings are excluded from benchmarks.

## SparkDash comparison

Five full decode and five prefill jobs ran through the user's SparkDash API.
All 295 decode streams completed 400 tokens each, with 399 timed decode tokens
and zero failures. All 100 measured prefill samples were uncached and valid;
25 warmup/calibration requests are excluded from that count.

Fresh controlled r6x and restored 0.32 R7 had decode medians within 0.25% at
every load. A return run at the original 10.5 GiB budget confirmed that result.
The final R8 results are consistent with that current-condition baseline.
The historical reference is one job, so it supplies no baseline variance.
There is no formal equivalence or causal speedup claim. Groups ran across
boots rather than request-by-request alternating.

Decode values are aggregate tok/s, median(range) across five jobs. C1 is
also the per-stream rate. The historical column contains one job.

| Clients | Fresh r6x | Final vLLM 0.32 | Historical single job |
| --- | ---: | ---: | ---: |
| 1 | 84.40 (84.22-85.03) | 84.82 (84.36-85.22) | 86.53 |
| 2 | 131.76 (122.79-135.34) | 133.56 (132.94-134.94) | 133.08 |
| 8 | 250.27 (248.89-250.75) | 250.89 (250.61-255.08) | 255.67 |
| 16 | 341.70 (341.19-341.99) | 343.92 (340.73-345.28) | 346.65 |
| 32 | 331.74 (330.10-334.68) | 337.37 (333.86-339.16) | 340.97 |

Prefill is SparkDash's tokens/max(TTFT minus calibrated overhead, 50% of
TTFT), not an isolated kernel rate. Overhead subtraction is capped at 50%
of TTFT; the historical 1K result uses that cap. Short 1K/2K historical results are marked low confidence.
Their overhead calibration was 207.14 ms and differs from the new jobs.
Read actual TTFT alongside derived rates.

| Context | Final tok/s median(range) | Historical tok/s | Final median TTFT | Historical TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1K | 5789.01 (5397.30-6398.13) | 6416.23 | 0.327s | 0.326s |
| 2K | 5746.10 (5673.84-6251.89) | 6789.13 | 0.505s | 0.511s |
| 4K | 6197.10 (6022.70-6422.30) | 6304.05 | 0.809s | 0.861s |
| 8K | 6311.44 (6258.34-6489.89) | 6445.43 | 1.446s | 1.481s |
| 16K | 6462.35 (6434.97-6526.09) | 6466.23 | 2.687s | 2.744s |
| 32K | 6443.45 (6399.31-6489.29) | 6428.31 | 5.234s | 5.308s |
| 64K | 6340.55 (6164.56-6358.95) | 6365.39 | 10.507s | 10.506s |

The C1 prompt is unchanged. Concurrent prompt revision equivalence to the
historical job is not established. The fresh r6x and final 0.32 jobs use the
same current SparkDash requests. The full 400-token C1 response matches
exactly across fresh r6x, R7, R8 and the R8 logprob fallback.

Final decode job IDs: `2fcbc86d-b327-43c2-98ae-77d75d9d5c1c`, `9b3af28a-a884-4a00-a254-b7c0cdf0cb9f`, `9b0a350b-90bb-4132-aaeb-8373e52e1d22`, `47e5c150-cfbd-44a7-8c8c-7e91702f5680`, `64d9e064-95c2-4565-87a7-f75b300df42c`.

Final prefill job IDs: `3ac0af4c-6201-4c92-a0d2-9994483f6119`, `aa665b35-915b-4fe7-bf46-0a369e74ecb3`, `1700586e-cafe-4ea9-964d-d4bd3ae0ce77`, `d0c0f6de-4842-4f30-9b01-2596ff1822c0`, `8456d0c2-d47b-4849-932a-f8e2f19e4570`.

## Storage and networking

Only spark3 stores the full pinned checkpoint under
`~/.local/share/deepseek1017-native/<revision>/checkpoint`. Existing read-only
Docker volumes share it over QSFP for loading. The native NFS server, export,
volumes and mounts remain unchanged. Do not restart NFS for serving changes.
The server binds [site/nfs/entrypoint.sh](site/nfs/entrypoint.sh) from this
folder. [NFS configuration](site/nfs/README.md) defines its retained resources.

Runtime Engram uses verified local ext4 O_DIRECT files on every rank under
`~/.local/share/deepseek1017-vllm/engram-local`. Never restore runtime Engram
on NFS. Writable caches use `~/.local/share/deepseek1017-vllm/cache-v032-fp4`.
Private deployment mirrors remain in their existing user-data directories.

QSFP carries runtime coordination and checkpoint reads. Master:
`10.100.227.2:25126`; socket interface `enp1s0f0np0`, GID3, MTU9000.
Preserve addresses, routes and metrics. SSH uses explicit Wi-Fi IPs.
NCCL retains paired PCI domains, 2 channels per net peer and library SHA256
`1bfbeb23d3ed1383fe8c77d91c12444d9dd51731b6dba29d63dcfe4e6b0a3ade`.

## Host and lifecycle

All four retain DGX OS 7.6.0, Ubuntu-signed 7.0.0-1019-nvidia-64k,
65536-byte pages and stock NVIDIA 580.178.04. Secure Boot stays enabled.
Memory Saver remains deferred. Protected guards, narrow sudo helpers,
host policy, firmware fan control and GLM resources remain unchanged.

The stable launcher `~/.local/libexec/llama-swap/deepseek-v41-1017-vllm`
uses [site/managed-launcher.sh](site/managed-launcher.sh) and the retained
shared lock. Do not start a second launcher. Keep API unload, independent
GLM/vLLM cleanup hooks and the 300-second stop budget. The GLM wrapper stops
vLLM before loading GLM. The retired native launcher/hook remain removed.

Per-request metrics and forced usage feed router Prefill/Decode rows.
These are scheduler timing rates. Global admission stays 32; DeepSeek and
GLM per-model limits stay 16. The model and router remain loaded.
