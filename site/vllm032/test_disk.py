from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vllm032.disk import DiskRows, dequantize, restore_heads


class DiskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sizes = [3] * 24
        self.offsets = np.arange(24, dtype=np.int64) * 3
        self.ids = self.offsets[None, :] + np.arange(5)[:, None] % 3
        values = (torch.arange(72) % 9 + 1).float().to(torch.float8_e4m3fn)
        raw = values.view(torch.uint8).numpy()[:, None].repeat(256, axis=1)
        self.paths = [self.root / 'values', self.root / 'scales']
        for path, data in zip(self.paths, [raw, np.full((72, 8), 127, dtype=np.uint8)]):
            payload = bytes(4096) + data.tobytes()
            path.write_bytes(payload + bytes((-len(payload)) % 4096))

    def tearDown(self):
        self.temp.cleanup()

    def reader(self, rank, **kwargs):
        return DiskRows(table_rows=72, head_sizes=self.sizes, rank=rank,
                        tp_size=4, max_tokens=8,
                        sources=[(self.paths[0], 4096, False), (self.paths[1], 4096, True)], **kwargs)

    def test_all_rank_columns_and_repeated_steps(self):
        outputs = []
        for rank in range(4):
            reader = self.reader(rank)
            try:
                result = reader.lookup(self.ids)
                expected = torch.tensor((self.ids[:, rank::4] % 9 + 1), dtype=torch.bfloat16)
                torch.testing.assert_close(result, expected.unsqueeze(-1).expand(-1, -1, 256), rtol=0, atol=0)
                smaller = reader.lookup(self.ids[:2])
                torch.testing.assert_close(smaller, result[:2], rtol=0, atol=0)
                outputs.append(result)
            finally:
                reader.close()
        gathered = restore_heads(torch.cat(outputs, dim=1), 4)
        expected = torch.tensor(self.ids % 9 + 1, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, -1, 256)
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

    def test_padding_empty_and_wrong_bucket(self):
        reader = self.reader(0)
        try:
            reader.lookup(self.ids)
            self.assertEqual(torch.count_nonzero(reader.lookup(np.full((2, 24), -1))).item(), 0)
            self.assertEqual(reader.lookup(np.empty((0, 24), dtype=np.int64)).shape, (0, 6, 256))
            invalid = self.ids.copy()
            invalid[0, 4] = 0
            with self.assertRaises(ValueError):
                reader.lookup(invalid)
        finally:
            reader.close()

    def test_short_read_fails(self):
        reader = self.reader(0)
        try:
            self.paths[0].write_bytes(bytes(4096))
            with self.assertRaises(RuntimeError):
                reader.lookup(self.ids)
        finally:
            reader.close()

    def test_timeout_is_terminal(self):
        reader = self.reader(0, timeout=0.01, read_delay=0.2)
        try:
            with self.assertRaises(TimeoutError):
                reader.lookup(self.ids)
            with self.assertRaises(RuntimeError):
                reader.lookup(self.ids[:1])
        finally:
            reader.close()

    def test_scale_extremes(self):
        values = torch.full((1, 256), 0x38, dtype=torch.uint8)
        scales = torch.tensor([[0, 1, 126, 127, 128, 254, 255, 127]], dtype=torch.uint8)
        result = dequantize(values, scales).reshape(8, 32)
        expected = torch.tensor([2.0**-127, 2.0**-126, 0.5, 1, 2, 2.0**127, float('nan'), 1], dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected[:, None].expand(-1, 32), rtol=0, atol=0, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
