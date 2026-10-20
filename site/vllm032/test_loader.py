import tempfile
from unittest import TestCase, mock

import torch

from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator


class LoaderTests(TestCase):
    def test_mixed_shard_does_not_materialize_excluded_table(self):
        table = 'layers.1.engram.embed.weight'
        projection = 'layers.1.engram.wkv.weight'
        tensor = torch.ones(2)
        handle = mock.MagicMock()
        handle.__enter__.return_value = handle
        handle.keys.return_value = [table, projection]

        def get_tensor(name):
            self.assertEqual(name, projection)
            return tensor

        handle.get_tensor.side_effect = get_tensor
        with tempfile.NamedTemporaryFile() as shard, mock.patch(
                'vllm.model_executor.model_loader.weight_utils.safe_open', return_value=handle):
            result = list(safetensors_weights_iterator(
                [shard.name], False, 'lazy', is_unused_weight=lambda name: name == table))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], projection)
        self.assertIs(result[0][1], tensor)
