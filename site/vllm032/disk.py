import atexit
from pathlib import Path
import time

import numpy as np
import torch

from ._native import load


def dequantize(values, scales):
    exponent = scales.to(torch.int32)
    factor = (exponent << 23).view(torch.float32)
    factor = torch.where(exponent == 0, 2.0**-127, factor)
    factor = torch.where(exponent == 255, float('nan'), factor)
    values = values.view(torch.float8_e4m3fn).float().unflatten(-1, (8, 32))
    return (values * factor.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def restore_heads(rows, tp_size):
    tokens, heads, dim = rows.shape
    return rows.reshape(tokens, tp_size, heads // tp_size, dim).transpose(1, 2).reshape(tokens, heads, dim)


class DiskRows:
    def __init__(self, *, table_rows, head_sizes, rank, tp_size, max_tokens, sources, timeout=20.0, read_delay=0.0):
        if len(head_sizes) != 24 or tp_size != 4 or rank not in range(tp_size):
            raise ValueError('Disk Engram requires the verified 24-head TP4 layout')
        self.rank = rank
        self.tp_size = tp_size
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.offsets = np.cumsum([0, *head_sizes[:-1]], dtype=np.int64)
        self.ends = self.offsets + np.asarray(head_sizes, dtype=np.int64)
        if int(self.ends[-1]) > table_rows:
            raise ValueError('Engram head buckets exceed the table')
        self.native = load()
        self.ids = np.empty(max_tokens * 24, dtype=np.int64)
        self.values = np.zeros((max_tokens * 24, 256), dtype=np.uint8)
        self.scales = np.zeros((max_tokens * 24, 8), dtype=np.uint8)
        self.control = np.zeros(self.native.CTRL_WORDS, dtype=np.uint32)
        columns = sum(1 << column for column in range(rank, 24, tp_size))
        self.reader = self.native.reader(
            table_rows, 0, table_rows, columns, max_tokens * 24, 128,
            self.ids.ctypes.data, self.values.ctypes.data, self.scales.ctypes.data,
            self.control.ctypes.data, timeout, read_delay, False, False,
        )
        self.sequence = 0
        self.failure = None
        try:
            for path, offset, scale in sources:
                self.native.add(self.reader, str(path), int(offset), bool(scale))
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def lookup(self, indices):
        if self.failure is not None:
            raise RuntimeError("Engram reader failed earlier") from self.failure
        if self.reader is None:
            raise RuntimeError("Engram reader is closed")
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] != 24 or ids.shape[0] > self.max_tokens:
            raise ValueError('Invalid Engram lookup shape')
        tokens = ids.shape[0]
        if not tokens:
            return torch.empty((0, 6, 256), dtype=torch.bfloat16, device='cpu')
        if not np.all((ids == -1) | ((ids >= self.offsets) & (ids < self.ends))):
            raise ValueError('Engram IDs do not match their hash buckets')
        count = ids.size
        self.ids[:count] = ids.reshape(-1)
        self.values[:count].fill(0)
        self.scales[:count].fill(0)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF or 1
        try:
            self.native.submit(self.reader, self.sequence, count, self.timeout, self.ids[:count])
            deadline = time.monotonic() + self.timeout
            while self.control[self.native.CTRL_ROWS_SEQ] != self.sequence:
                status = self.native.status(self.reader)
                if status['failure'] is not None:
                    raise RuntimeError(status['failure'])
                if time.monotonic() >= deadline:
                    raise TimeoutError('Engram disk lookup timed out')
                time.sleep(0.0001)
            status = self.native.status(self.reader)
            if status['failure'] is not None:
                raise RuntimeError(status['failure'])
        except Exception as error:
            self.failure = error
            raise
        values = torch.from_numpy(self.values[:count].reshape(tokens, 24, 256)[:, self.rank::4].copy())
        scales = torch.from_numpy(self.scales[:count].reshape(tokens, 24, 8)[:, self.rank::4].copy())
        self.control[self.native.CTRL_CONSUMED_SEQ] = self.sequence
        result = dequantize(values, scales)
        valid = torch.from_numpy(ids[:, self.rank::4] != -1)
        return result.masked_fill_(~valid.unsqueeze(-1), 0)

    def close(self):
        self.reader = None
        atexit.unregister(self.close)


def from_local_table(table_rows, head_sizes, rank, tp_size, max_tokens):
    from local_engram import load_plan, local_source, read_manifest
    import os

    root = Path(os.environ['SPARK3_ENGRAM_LOCAL_DIR'])
    manifest = read_manifest(root)
    plan = load_plan()
    if manifest['rank'] != rank or manifest['tp_size'] != tp_size:
        raise ValueError('Local Engram rank mismatch')
    layer = plan['geometry']['num_embeddings'].index(table_rows)
    if list(head_sizes) != plan['geometry']['primes'][layer]:
        raise ValueError('Local Engram bucket geometry mismatch')
    sources = []
    columns = sum(1 << column for column in range(rank, 24, tp_size))
    for name, info in manifest['files'].items():
        for key, tensor in info['tensors'].items():
            if tensor['shape'][0] == table_rows:
                scale = key.endswith('.scale')
                path = local_source(root / name, tensor['absolute_offset'], scale, table_rows, columns)
                sources.append((path, tensor['absolute_offset'], scale))
    if len(sources) != 2:
        raise ValueError('Expected one Engram value tensor and one scale tensor')
    return DiskRows(table_rows=table_rows, head_sizes=head_sizes, rank=rank,
                    tp_size=tp_size, max_tokens=max_tokens, sources=sources)
