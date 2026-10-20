# Local runtime Engram

The site checkpoint was mounted over NFS on spark1/2/4 for initial loading. The native Engram reader also opened its random runtime rows there. This violated the intended storage boundary and made source-text prefill slow. NFS is now only for checkpoint loading and preparation; both Engram tables must read local disk on every rank.

Each rank owns six interleaved hash columns of both tables. Spark1/2/4 keep only those original weight and scale ranges, rounded outward to4KiB, in sparse files with the original logical sizes and offsets. This uses about47.21GiB per worker. Spark3 hardlinks its existing original files and uses no additional weight data. Original model files, precision, kernels, column exchange and fabric are preserved.

`local-plan.json` records actual headers and canonical B12X geometry for the pinned official checkpoint. `local_engram.py` checks its pinned SHA, exact file/tensor/range descriptors, rank ownership and local filesystems. Preparation hashes every copied range against its destination, syncs and atomically publishes the files and manifest. Repeated preparation verifies an existing completed copy. Partial files are rebuilt safely. Source/destination aliases are rejected. The managed launcher verifies every local range and overlay hash before starting any rank. Missing or inconsistent local data aborts startup; there is no runtime NFS fallback.

The read-only `rows.py` overlay substitutes only the source path passed to the unchanged native reader. A helper and the pinned plan are also mounted read-only. The original SGLang recipe and its packed Engram are preserved.

Validation status: CPU ownership, copied-byte, corruption, metadata, wrong-rank, source-alias and repeat-run tests passed. Live performance and numerical comparisons are pending. No capacity or full1M benchmark is required for this fix.
