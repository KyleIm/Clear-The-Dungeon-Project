import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rl.decision_stats import DecisionStats
from rl.reporting import ExperimentStore, generate_report


class DecisionStatisticsTests(unittest.TestCase):
    def test_checkpoint_rollback_removes_only_affected_decision_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'markers.sqlite3'
            with ExperimentStore(db, batch_size=1) as store:
                row = {'step': 8, 'stage': 1, 'split': 'validation', 'policy': 'ppo_stochastic',
                       'decision_stats': DecisionStats().summary()}
                store.evaluation(row)
                marker = store.watermarks()
                store.evaluation(row)
                store.restore_watermarks(marker)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM decision_stats').fetchone()[0], 1)
                payload = connection.execute('SELECT payload FROM evaluations_with_decisions').fetchone()[0]
                self.assertIn('decision_stats', json.loads(payload))
                self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_combat_counts_distinguish_availability_and_choice(self):
        stats = DecisionStats()
        stats.record(0, 1, {0: 'thunder', 1: 'counter'}, 1, [0.6, 0.4])
        stats.record('combat', 2, {0: 'thunder'}, 0, {0: 1.0})
        stats.record('combat', 2, {0: 'thunder'}, 0, None)
        combat = stats.summary()['combat']
        self.assertEqual(combat['thunder']['available'], 3)
        self.assertEqual(combat['thunder']['selected'], 2)
        self.assertAlmostEqual(combat['thunder']['selection_rate'], 2 / 3)
        self.assertAlmostEqual(combat['thunder']['mean_probability'], 0.8)
        self.assertEqual(combat['counter']['selection_rate'], 1)

    def test_reward_events_and_interleaved_shop_visits(self):
        stats = DecisionStats()
        stats.record('reward', 1, {0: 'Gold', 1: 'Potion'}, 0, None, slot=0)
        stats.record('reward', 3, {0: 'Gold', 1: 'Potion'}, 1, None, slot=1)
        stats.record('event', 35, {0: 'help', 1: 'pickpocket'}, 0, None, slot=0)
        for slot, choice in [(0, 0), (1, 0), (0, 0), (0, 1)]:
            stats.record('shop', 40, {0: 'Skill Configuration', 1: 'Leave Shop'}, choice, [0.8, 0.2], slot=slot)
        stats.transition({'shop_selection': {'function': 'Skill Configuration', 'repeat': False}}, slot=0)
        stats.transition({'shop_selection': {'function': 'Skill Configuration', 'repeat': False}}, slot=1)
        stats.transition({'shop_selection': {'function': 'Skill Configuration', 'repeat': True},
                          'shop_penalty': -0.0001}, slot=0)
        stats.transition({'shop_visit': {'strategic_actions': 3, 'engine_actions': 5, 'left': True,
                                        'repeated': True, 'penalty': -0.0001, 'outcome': None}}, slot=0)
        stats.transition({'shop_visit': {'strategic_actions': 1, 'engine_actions': 7, 'left': False,
                                        'repeated': False, 'penalty': 0, 'outcome': 'timeout'}}, slot=1)
        stats.episode({'floor': 100, 'outcome': 'clear'}, slot=0)
        stats.reset_slot(0)
        stats.episode({'floor': 40, 'outcome': 'timeout'}, slot=1)
        result = stats.summary()
        self.assertEqual(result['reward']['by_floor']['1']['Gold'], 1)
        self.assertEqual(result['reward']['rates']['Potion'], 0.5)
        self.assertEqual(result['reward']['available_by_floor']['1']['Potion'], 1)
        self.assertEqual(result['reward']['selection_rates']['Potion'], 0.5)
        self.assertEqual(result['event']['options']['help']['clear_rate'], 1)
        self.assertEqual(result['event']['options']['help']['mean_floor'], 100)
        self.assertEqual(result['event']['options']['pickpocket']['available'], 1)
        self.assertEqual(result['event']['options']['pickpocket']['selection_rate'], 0)
        self.assertEqual(result['shop']['functions']['Skill Configuration']['first'], 2)
        self.assertEqual(result['shop']['functions']['Skill Configuration']['repeat'], 1)
        self.assertEqual(result['shop']['functions']['Leave Shop']['selected'], 1)
        self.assertEqual(result['shop']['functions']['Leave Shop']['available'], 4)
        self.assertEqual(result['shop']['mean_engine_actions'], 6)
        self.assertEqual(result['shop']['exit_rate'], 0.5)
        self.assertEqual(result['shop']['timeout_rate'], 0.5)
        self.assertAlmostEqual(result['shop']['penalty'], -0.0001)
        self.assertAlmostEqual(result['shop']['mean_leave_probability'], 0.2)

    def test_report_contains_decision_tables_and_bounded_shop_examples(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('Matplotlib unavailable')
        stats = DecisionStats()
        stats.record('combat', 1, {0: 'thunder'}, 0, [1.0])
        stats.record('reward', 1, {0: 'Gold'}, 0, [1.0])
        for count in range(12):
            stats.transition({'shop_change': {'function': 'Skill Configuration', 'before': {'mana_used': 12},
                                              'after': {'mana_used': 13 + count}, 'gold_spent': 0}})
        result = stats.summary()
        json.dumps(result, allow_nan=False)
        skill = result['shop']['changes']['Skill Configuration']
        self.assertEqual(skill['commits'], 12)
        self.assertLess(len(skill['examples']), 12)
        self.assertEqual(skill['examples'][0]['before']['mana_used'], 12)
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'statistics.sqlite3'
            with ExperimentStore(db) as store:
                for policy in ('ppo_deterministic', 'ppo_stochastic'):
                    store.evaluation({'step': 100, 'stage': 1, 'split': 'validation', 'policy': policy,
                                      'episodes': 1, 'decision_stats': result})
            report = generate_report(db, Path(temp) / 'report')
            self.assertIn('decision_statistics.md', report.read_text(encoding='utf-8'))
            detailed = report.with_name('decision_statistics.md').read_text(encoding='utf-8')
            self.assertIn('thunder', detailed)
            self.assertIn('Gold', detailed)
            self.assertIn('Skill Configuration', detailed)
            self.assertIn('stochastic', detailed)
            self.assertTrue(report.with_name('decision_trends.png').is_file())
