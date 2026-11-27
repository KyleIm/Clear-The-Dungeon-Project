from contextlib import closing
from importlib.util import find_spec
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(find_spec('numpy'), 'NumPy required')
class AlternatingReportTests(unittest.TestCase):
    def test_recent_training_samples_exclude_evaluation_and_censoring_and_restore(self):
        import numpy as np

        from alternating_rl.report import Store

        with TemporaryDirectory() as tmp, Store(Path(tmp) / 'run.sqlite3', batch_size=100) as store:
            features = np.arange(671, dtype=np.float32)
            base = dict(features=features, shop_floor=10, cycle=1, prediction=.2)
            for split, version, target in [('train', 2, 0), ('train', 1, 1), ('eval', 9, 1),
                                           ('train', 3, None), ('train', 4, 1)]:
                store.sample(split, 10, dict(base, version=version, target=target))
            rows = store.recent_samples(10, current_version=3)
            self.assertEqual([(row['version'], row['target']) for row in rows], [(2, 0), (1, 1)])
            np.testing.assert_array_equal(rows[0]['features'], features)
            saved = store.marks()
            store.sample('train', 10, dict(base, version=3, target=1))
            self.assertEqual(store.recent_samples(1, current_version=3)[0]['version'], 3)
            store.restore(saved)
            self.assertEqual(store.marks(), saved)
            self.assertEqual(store.recent_samples(1, current_version=3)[0]['version'], 2)

    @unittest.skipUnless(find_spec('matplotlib'), 'Matplotlib required')
    def test_report_combines_episode_shop_and_paired_results(self):
        import numpy as np

        from alternating_rl.report import Store, generate_report

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'run.sqlite3'
            with Store(path) as store:
                row = dict(game_seed=4, policy_seed=8, floor=42, actions=50, outcome='game_over',
                           detail=None, cycle=1, version=2, segment=2, combat_actions={'slot_1': 8},
                           skill_actions={'thunder': 8}, skill_casts={'thunder': 6}, reward_actions={'Gold': 2},
                           event_route='B', special_shop_entered=False, death_monster='Mimic')
                entry = dict(floor=10, gold=1000, unused_stat_points=20, remaining_mana=39, unlocked_slots=4)
                exit_state = dict(entry, gold=300, unused_stat_points=0, candidates=8, explored=True,
                                  slots=[dict(skill='thunder', uses=4)], actions=[dict(action=['buy', 'mana_potion'])])
                for split in ('train', 'eval'):
                    store.episode(split, 10, dict(row, shop_entries=[entry], shop_exits=[exit_state]))
                    store.sample(split, 10, dict(features=np.zeros(671), shop_floor=10, cycle=1, version=2,
                                                 prediction=.8, target=1))
                store.update(10, dict(shop_loss=.5, episodes_per_second=25.0))
                store.evaluation(10, 'eval', dict(seconds=1.5))
            report = generate_report(path, root / 'report')
            text = report.read_text(encoding='utf-8')
            for phrase in ('Cycle 1 / segment 2', 'Entries 1. Exits 1. Success/failure/censored 1/0/0',
                           'buy mana_potion', 'F99 shop entries', 'Resolved player skill casts', 'thunder',
                           'Historical PPO 100K', 'Historical PPO 1M', '0.8–0.9', 'episodes_per_second'):
                self.assertIn(phrase, text)
            for name in ('progress.png', 'decision_trends.png'):
                self.assertGreater((report.parent / name).stat().st_size, 1000)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT length(features) FROM shop_samples').fetchall(), [(2684,), (2684,)])


if __name__ == '__main__':
    unittest.main()
