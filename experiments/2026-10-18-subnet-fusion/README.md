# Peer-aware physical-port pairs

Base: pinned NCCL 2.30.7 plus the promoted adaptive patch series, commit eeacf1c6343f0499e4e7ab20ec7eecfb6f62eda6. Apply `series` in order in a fresh named source checkout. The second candidate is qualified for this site with the paired-P2P2 profile. It has not been promoted upstream or submitted upstream.

Forced physical-port pairs fail when the listener advertises only the selected pair, which can face another neighbor. Advertise one representative per enabled cable subnet in the existing two-slot handle, then select the largest device whose every member reaches the peer. Reject excess subnet counts and incomplete member queries. The second patch selects the receiver before device-count and capability negotiation, sends selected-member GIDs in the initial properties exchange and persists the connector selection across asynchronous polls. The opaque handle is104bytes, below its128-byte limit. All ranks must use the same library: its private handshake format differs from the base. Both tested forced groups have exactly two members. GPU kernels, arithmetic, checkpoint, IP addresses, routes, MTU and the one-shot peer map are unchanged.

Build a separate SM121 NCCL library from the exact source base and patch. Keep the serving image and native/GLM libraries intact. Test only in owned model-free probe containers first. Require original BF16/FP32 correctness and graph replay, all-rank zero RDMA errors, two-member connection proof and actual vport payload on both PCI domains. No speed or quality claim exists before those gates and matched serving requests pass.


The first candidate rejected the primary control before timing: a receiver tried to change from an auto-merged two-member device to a single reachable member after negotiating counts. Its logs and original library remain under `performance-replication/subnet-*`; the second candidate uses distinct `subnet-v2-*` artifacts. No timing claim comes from the rejected candidate.

Site qualification passed at signed8d004a9: five source32K requests median5760.11tok/s and five capped hash-map prose requests81.63tok/s, actual all-rank library/pair/payload proof, local runtime Engram/NFS READ0 and basic APIs/rates. Scope and immutable evidence: `manifests/baselines/2026-10-18-sparks-local-engram-paired-p2p2.json`. This does not qualify the full upstream suite or every NCCL topology.

Published patch mail uses normalized author metadata. `base.json` retains the
historical candidate identities and records the published source head and tree
separately. An unchanged source tree does not establish an identical rebuilt
library hash; the qualified installed library remains checked by
`site/check-nccl-library.py`.
