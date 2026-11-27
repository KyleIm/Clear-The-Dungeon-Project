import json
from pathlib import Path
import unittest
from unittest.mock import patch

from combat_rl.performance import Profile, Resources


class PerformanceTests(unittest.TestCase):
    def test_profile_records_inclusive_elapsed_and_count(self):
        profile = Profile(detailed=True)
        profile.add('worker_round', 0.5)
        with patch('combat_rl.performance.perf_counter', side_effect=(4, 5)):
            with profile.measure('worker_round'):
                pass
        self.assertTrue(profile.detailed)
        self.assertEqual(profile.summary(), {'worker_round': {'seconds': 1.5, 'calls': 2}})

    def test_gpu_query_has_timeout_and_maps_unavailable_fields(self):
        resources = Resources(interval=0.1)
        resources._gpu_command = Path('nvidia-smi')
        with patch('combat_rl.performance.subprocess.run') as run:
            run.return_value.stdout = 'NVIDIA Test, [N/A], 123\n'
            self.assertEqual(resources._gpu(), ('NVIDIA Test', None, 123.0))
            self.assertGreater(run.call_args.kwargs['timeout'], 0)

    def test_local_process_samples_and_json_output(self):
        with patch('combat_rl.performance.shutil.which', return_value=None):
            with Resources(interval=0.05) as resources:
                self.assertTrue(resources._ready.wait(timeout=3))
        summary = resources.summary()
        self.assertGreaterEqual(summary['samples'], 1)
        self.assertIsNone(summary['gpu_util_percent_mean'])
        self.assertIsNone(summary['gpu_memory_used_mib_peak'])
        if resources._psutil is not None:
            self.assertGreater(summary['process_tree_rss_mib_peak'], 0)
            self.assertGreater(summary['system_ram_used_mib_peak'], 0)
        json.dumps(summary, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
