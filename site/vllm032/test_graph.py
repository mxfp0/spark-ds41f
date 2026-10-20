def main():
    import os
    os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '1'
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import torch
    from torch import nn
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm032.embedding import DiskEmbedding
    from vllm032.test_disk import DiskTests

    fixture = DiskTests()
    fixture.setUp()
    try:
        for rank in range(4):
            module = DiskEmbedding.__new__(DiskEmbedding)
            nn.Module.__init__(module)
            module.disk = fixture.reader(rank)
            lookup = DiskEmbedding.lookup
            for tokens in [1, 4]:
                ids = torch.tensor(fixture.ids[:tokens], device='cuda')
                produced = torch.full_like(ids, -2)
                out = torch.empty((tokens, 6, 256), device='cuda', dtype=torch.bfloat16)
                stream = torch.cuda.Stream()
                torch.cuda.synchronize()
                with patch('vllm032.embedding.get_forward_context', return_value=SimpleNamespace(attn_metadata={})):
                    with torch.cuda.stream(stream):
                        lookup(module, ids, out)
                        sequence = module.disk.sequence
                        cap = BreakableCUDAGraphCapture()
                        with cap:
                            torch.add(ids, 0, out=produced)
                            lookup(module, produced, out)
                            result = out + 1
                        assert cap.num_eager_breaks == 1
                        assert module.disk.sequence == sequence
                        for offset in [2, 0, 1, -1]:
                            changed = fixture.offsets[None, :] + np.full((tokens, 24), offset)
                            if offset == -1:
                                changed.fill(-1)
                            ids.copy_(torch.tensor(changed, device='cuda'))
                            sequence = module.disk.sequence
                            cap.replay()
                            torch.cuda.synchronize()
                            assert module.disk.sequence == sequence + 1
                            expected = (torch.tensor(changed[:, rank::4] % 9 + 1, dtype=torch.bfloat16) if offset != -1 else torch.zeros((tokens, 6), dtype=torch.bfloat16)) + 1
                            torch.testing.assert_close(result.cpu(), expected.unsqueeze(-1).expand(-1, -1, 256), rtol=0, atol=0)
                        ids.fill_(-2)
                        sequence = module.disk.sequence
                        try:
                            cap.replay()
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('Invalid replay IDs were accepted')
                        assert module.disk.sequence == sequence
                print('Fresh disk rows on graph replay', rank, tokens, flush=True)
            module.disk.close()
    finally:
        fixture.tearDown()


if __name__ == "__main__":
    main()
