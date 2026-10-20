import importlib.machinery
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_site_test", str(ROOT / "bin/spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class SiteSetupTest(unittest.TestCase):
    def test_base_image_sources_require_explicit_mode(self):
        sources = spark.read_json("site/vllm032/upstreams.lock.json")["sources"]
        self.assertEqual(spark.build_projects(sources), ("vllm", "tilelang", "tile_kernels", "sparknet"))
        for flag in (False, "true", None):
            sources["vllm"]["runtime_from_base_image"] = flag
            with self.assertRaisesRegex(SystemExit, "required build source b12x"):
                spark.build_projects(sources)
        del sources["vllm"]
        with self.assertRaisesRegex(SystemExit, "required build source vllm"):
            spark.build_projects(sources)

    def test_guard_helpers_preserve_system_scope_defaults(self):
        node = {"name": "spark1"}
        cluster = {"container": {"name": "model"}, "host": {}}
        with patch.object(spark, "run_ssh") as run:
            run.return_value.returncode = 0
            spark.stop_memguard(cluster, {}, node)
            self.assertEqual(run.call_args.args[2:], ("sudo", "-n", "systemctl", "stop", "model-memguard.service"))
            self.assertTrue(spark.memguard_is_active(cluster, {}, node))
            self.assertEqual(run.call_args.args[2:], ("systemctl", "is-active", "--quiet", "model-memguard.service"))
            helper = "/usr/local/libexec/deepseek1017/memguard-control"
            cluster["host"]["memguard_helper"] = helper
            spark.stop_memguard(cluster, {}, node)
            self.assertEqual(run.call_args.args[2:], ("sudo", "-n", helper, "stop"))
            self.assertTrue(spark.memguard_is_active(cluster, {}, node))
            self.assertEqual(run.call_args.args[2:], ("sudo", "-n", helper, "active"))

    def test_installed_site_guard_keeps_user_and_oom_protection(self):
        source = (ROOT / "site/host/install.sh").read_text()
        body = re.search(r"<<'PY'\n(.*?)\nPY\n", source, re.S).group(1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memguard-start.sh"
            subprocess.run([sys.executable, "-c", body, str(ROOT / "scripts/memguard-start.sh"), str(path)], check=True)
            subprocess.run(["bash", "-n", str(path)], check=True)
            script = path.read_text()
        self.assertEqual(script.count("--property=User=spark"), 1)
        self.assertEqual(script.count("--property=OOMScoreAdjust=-1000"), 1)
        self.assertIn("repo_root=/usr/local/libexec/deepseek1017", script)
        self.assertIn('"$repo_root/memguard.sh"', script)
        self.assertNotIn("sudo -n", script)
        self.assertNotIn("--user", script)


if __name__ == "__main__":
    unittest.main()
