import argparse
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPARK = runpy.run_path(str(ROOT / 'bin/spark'))
POLICY = SPARK['kernel_backend']


class UpstreamBackendTests(unittest.TestCase):
    def setUp(self):
        self.cluster, _, self.lock = SPARK['configuration'](
            argparse.Namespace(cluster_config='config/sparks.json'))
        self.manifest = SPARK['read_json'](self.lock['source_manifest'])

    def test_site_has_matching_backend_and_sources(self):
        self.assertEqual(POLICY.problems(self.cluster), [])
        self.assertEqual(POLICY.source_problems(self.cluster, self.lock, self.manifest), [])

    def test_unmatched_image_tree_is_rejected(self):
        self.cluster['container']['expected_labels']['local.spark3.vllm.tree'] = '0' * 40
        errors = POLICY.source_problems(self.cluster, self.lock, self.manifest)
        self.assertTrue(any('image tree' in error for error in errors), errors)

    def test_missing_native_and_legacy_sources_are_rejected(self):
        del self.lock['sources']['vllm']
        self.assertTrue(any('requires source vllm' in error for error in
                            POLICY.source_problems(self.cluster, self.lock, self.manifest)))
        legacy = {'kernel_backend': 'b12x'}
        errors = POLICY.source_problems(legacy, {'sources': {'vllm': {}}}, {})
        self.assertTrue(any('requires source b12x' in error for error in errors), errors)
        self.assertTrue(any('requires source nccl' in error for error in errors), errors)

    def test_wrong_backend_environment_is_rejected(self):
        self.cluster['environment'][POLICY.ENVIRONMENT] = 'b12x'
        self.assertTrue(POLICY.problems(self.cluster))

    def test_native_profile_rejects_missing_cache_capability(self):
        self.manifest['vllm']['capabilities'].remove('native-fp4-cache')
        self.assertTrue(any('native-fp4-cache' in error for error in
                            POLICY.source_problems(self.cluster, self.lock, self.manifest)))

    def test_native_profile_rejects_another_attention_format(self):
        args = self.cluster['serve_args']
        args[args.index('--attention-backend') + 1] = 'FLASHINFER_MLA_SPARSE_DSV41'
        self.assertTrue(POLICY.problems(self.cluster))


if __name__ == '__main__':
    unittest.main()
