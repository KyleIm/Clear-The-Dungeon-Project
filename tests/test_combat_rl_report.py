import importlib.util
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest


class CombatReportTests(unittest.TestCase):
    def test_store_checkpoint_rollback(self):
        from combat_rl.report import Store

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runs.sqlite3'
            with Store(path) as db:
                sample = dict(game_seed=2**64 - 1, policy_seed=2, floor=4, actions=18,
                              outcome='game_over', detail=None)
                db.episode('train', 10, sample)
                db.update(10, {'episodes': 10, 'policy_loss': .2})
                marks = db.marks()
                db.episode('eval', 10, sample)
                db.evaluation(10, 'eval', {'elapsed': .1})
                db.restore(marks)
            db = sqlite3.connect(path)
            try:
                self.assertEqual(db.execute('SELECT split, game_seed FROM episodes').fetchall(),
                                 [('train', str(2**64 - 1))])
                self.assertEqual(db.execute('SELECT count(*) FROM updates').fetchone(), (1,))
                self.assertEqual(db.execute('SELECT count(*) FROM evaluations').fetchone(), (0,))
            finally:
                db.close()

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable')
    def test_report_reduces_telemetry_and_separates_policies(self):
        from combat_rl.report import Store, generate_report

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'runs.sqlite3'
            sample = dict(game_seed=1, policy_seed=2, floor=12, actions=26, outcome='game_over',
                          death_monster='Rogue test monster', combat_actions={'slot_1': 10},
                          reward_actions={'Gold': 2}, shop_actions=[{'floor': 10, 'action': ['leave_shop']}],
                          shop_exits=[{'unused_stat_points': 4, 'remaining_mana': 33}])
            with Store(path) as db:
                db.episode('train', 10, sample)
                db.episode('eval', 10, sample)
                db.episode('random', 10, dict(sample, floor=9, outcome='timeout', detail='action_limit',
                                              timeout_phase='combat'))
                db.update(10, {'episodes': 10, 'policy_loss': .2, 'value_loss': .3})
            result = generate_report(path, root / 'report')
            report = result.read_text(encoding='utf-8')
            for text in ('1,000,000', 'slot_1', 'leave_shop', '4.00', '33.00',
                         'Rogue test monster', 'action_limit', 'Paired uniform random'):
                self.assertIn(text, report)
            self.assertTrue((result.parent / 'floor_reach.png').is_file())
            self.assertTrue((result.parent / 'learning_curves.png').is_file())
