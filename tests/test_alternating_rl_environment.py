import importlib.util
from random import Random
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is required for RL features')
class AlternatingEnvironmentTests(unittest.TestCase):
    def make_run(self, *, cap=100, random_shop=True):
        from alternating_rl.environment import Run

        return Run(24, 32, cap, 3, 5, random_shop=random_shop)

    def show_reward(self, run, floor):
        run.game.floor = run.floor = floor
        run.game._offer_reward()
        run.state = run.game.get_state()

    def leave_random_shop(self, run):
        run.policy_rng = Random(5)  # The first draw selects the fifth entry, Leave Shop.
        self.assertEqual(run.choose_random_shop(), 1.0)

    def defeat_enemy(self, run):
        def win(game, *_):
            game.monster.hp = 0

        with patch('simulator.game.combat.resolve_turn', side_effect=win):
            return run.step(run._decision_actions()[0][0])

    def test_bootstrap_random_shop_matches_historical_replay(self):
        from alternating_rl.environment import Run

        run = Run(2538817474026044487, 3909549678122254972, 100, 3, 5, random_shop=True)
        while not run.done:
            if run.phase == 'shop':
                run.choose_random_shop()
            else:
                run.step(run.random_action())
        result = run.result()
        self.assertEqual((result['floor'], result['actions'], result['outcome']), (12, 83, 'game_over'))
        self.assertEqual(result['reward'], 11)
        self.assertEqual(len(result['shop_entries']), 1)
        self.assertNotEqual(result['shop_entries'][0]['region'], 'Shop')
        self.assertEqual(sum(result['skill_actions'].values()), sum(result['combat_actions'].values()))
        self.assertGreater(sum(result['skill_casts'].values()), 0)
        sample, = run.drain_samples()
        self.assertEqual((sample['target'], sample['reason'], sample['version'], sample['cycle']),
                         (0, 'game_over', 5, 3))
        self.assertNotIn('features', sample)
        self.assertFalse(run.drain_samples())

    def test_label_is_emitted_when_the_next_shop_opens(self):
        run = self.make_run()
        self.show_reward(run, 10)
        self.assertEqual(run.step(0), 0)
        self.leave_random_shop(run)
        self.assertFalse(run.drain_samples())
        run.game.floor = run.floor = 20
        run.state = run.game.get_state()
        run.version = 6
        self.assertFalse(run.drain_samples())
        self.defeat_enemy(run)
        self.assertFalse(run.drain_samples())
        run.step(0)
        sample, = run.drain_samples()
        self.assertEqual((sample['shop_floor'], sample['target'], sample['reason']), (10, 1, 'milestone'))
        self.assertEqual((sample['version'], sample['resolution_version']), (5, 6))
        self.assertEqual(run.phase, 'shop')
        self.assertEqual(run.shop_exits[0]['target'], 1)

    def test_at_cap_an_unresolved_exit_is_censored(self):
        run = self.make_run(cap=2)
        self.show_reward(run, 10)
        run.step(0)
        self.leave_random_shop(run)
        sample, = run.drain_samples()
        self.assertIsNone(sample['target'])
        self.assertEqual(sample['reason'], 'action_limit')
        self.assertEqual(run.result()['outcome'], 'timeout')

    def test_milestone_on_final_allowed_action_is_a_successful_label(self):
        run = self.make_run(cap=4)
        self.show_reward(run, 10)
        run.step(0)
        self.leave_random_shop(run)
        run.game.floor = run.floor = 20
        run.state = run.game.get_state()
        self.defeat_enemy(run)
        run.step(0)
        self.assertEqual(run.result()['outcome'], 'timeout')
        sample, = run.drain_samples()
        self.assertEqual((sample['target'], sample['reason']), (1, 'milestone'))

    def test_ordered_shop_keeps_terminal_transition_pending_until_milestone(self):
        run = self.make_run(random_shop=False)
        self.show_reward(run, 10)
        run.step(0)
        rng = run.game.rng.getstate()
        for expected_head in range(3):
            features, head, mask = run.shop_observation()
            self.assertEqual(features.shape, (688,))
            self.assertEqual(head, expected_head)
            reward = run.choose_shop(len(mask) - 1, .25, False)
            self.assertEqual(reward, 1.0 if head == 2 else 0.0)
        self.assertNotEqual(run.game.rng.getstate(), rng)  # Entering the next floor consumes RNG.
        rows = run.drain_transitions()
        self.assertEqual([row['next_state'][1] for row in rows], [1, 2])
        self.assertFalse(run.drain_samples())
        run.game.floor = run.floor = 20
        run.state = run.game.get_state()
        self.defeat_enemy(run)
        run.step(0)
        terminal, = run.drain_transitions()
        self.assertTrue(terminal['terminal'])
        self.assertEqual(terminal['reward'], 1.0)
        self.assertIsNone(terminal['next_state'])
        self.assertEqual(run.drain_samples()[0]['exit_q'], .25)

    def test_uniform_immediate_leave_produces_compatible_ordered_edges(self):
        from alternating_rl.environment import Run

        run = Run(24, 32, 100, 1, 0, random_shop=True, collect_uniform=True)
        self.show_reward(run, 10)
        run.step(0)
        self.leave_random_shop(run)
        self.assertTrue(run.shop_exits[-1]['uniform_replay_compatible'])
        rows = run.drain_transitions()
        self.assertEqual([row['state'][1] for row in rows], [0, 1])
        self.assertTrue(all(row['source'] == 'uniform' for row in rows))

    def test_uniform_trace_with_stage_regression_is_rejected(self):
        from alternating_rl.bootstrap import compatible_trace, shadow_shop
        from alternating_rl.shop import BUDGETS

        run = self.make_run()
        self.show_reward(run, 10)
        run.step(0)
        before = run.game.get_state()
        raw = [('shop_menu', 'skills'), ('shop_back',), ('shop_menu', 'stats'), ('shop_back',)]
        self.assertIsNone(compatible_trace(shadow_shop(run.game), run.game, raw, 'Forest', None, BUDGETS))
        self.assertEqual(run.game.get_state(), before)

    def test_checkpoint_preserves_pending_label_and_random_stream(self):
        import pickle

        from alternating_rl.environment import Run

        run = Run(2538817474026044487, 3909549678122254972, 100, 3, 5, random_shop=True)
        while run.phase != 'shop':
            run.step(run.random_action())
        run.choose_random_shop()
        self.assertFalse(run.drain_samples())
        restored = pickle.loads(pickle.dumps(run))
        for candidate in (run, restored):
            while not candidate.done:
                candidate.step(candidate.random_action())
        self.assertEqual(run.result(), restored.result())
        self.assertEqual(run.drain_samples()[0]['reason'], restored.drain_samples()[0]['reason'])

    def test_floor_99_routes_and_horizon_success(self):
        for event_route, path in (('A', 1), ('B', 2)):
            with self.subTest(route=event_route):
                run = self.make_run()
                self.show_reward(run, 90)
                run.event_route = event_route
                run.game.ella_path = path
                run.game.gate_queue = [area['boss_id'] for area in run.game.data.regions.values()
                                       if 'boss_id' in area]
                self.assertEqual(len(run.game.gate_queue), 9)
                run.state = run.game.get_state()
                run.step(0)
                self.leave_random_shop(run)
                for floor in range(91, 100):
                    self.assertEqual(run.floor, floor)
                    self.assertEqual(run.phase, 'combat')
                    self.defeat_enemy(run)
                    run.step(0)
                self.assertEqual(run.floor, 99)
                self.assertFalse(run.drain_samples())
                if event_route == 'A':
                    self.assertEqual(run.phase, 'combat')
                    self.assertEqual(run.state['monster']['name'], 'Ella')
                    self.defeat_enemy(run)
                    run.step(0)
                else:
                    self.assertEqual(run.phase, 'shop')
                    self.leave_random_shop(run)
                sample, = run.drain_samples()
                self.assertEqual((sample['shop_floor'], sample['target']), (90, 1))
                self.assertEqual((run.floor, run.phase), (100, 'combat'))
                self.defeat_enemy(run)
                if event_route == 'A':
                    self.assertFalse(run.drain_samples())
                else:
                    sample, = run.drain_samples()
                    self.assertEqual((sample['shop_floor'], sample['target']), (99, 1))
                result = run.result()
                self.assertEqual(result['outcome'], 'cleared')
                self.assertEqual(result['special_shop_entered'], event_route == 'B')
                self.assertEqual(result['special_shop_exited'], event_route == 'B')

    def test_event_choice_is_recorded_and_uses_a_single_uniform_draw(self):
        for seed, route in ((1, 'A'), (5, 'B')):
            with self.subTest(seed=seed):
                run = self.make_run()
                run.game.floor = run.floor = 35
                run.state = run.game.get_state()
                run.policy_rng = Random(seed)
                expected_rng = Random(seed)
                expected_rng.choice([('ella', 'pickpocket'), ('ella', 'help')])
                self.defeat_enemy(run)
                self.assertEqual(run.phase, 'reward')
                self.assertEqual(run.event_route, route)
                self.assertEqual(run.event_version, run.version)
                self.assertEqual(run.policy_rng.getstate(), expected_rng.getstate())
