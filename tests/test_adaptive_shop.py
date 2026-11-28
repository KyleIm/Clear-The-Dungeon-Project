import unittest
from unittest.mock import patch

import numpy as np
import torch

from adaptive_shop.reward import mixture_probability, guide
from adaptive_shop.promotion import decision
from adaptive_shop.gameplay import Run, planning
from adaptive_shop.runner import mode
from baseline_shop.configurations import Candidate
from combat_rl.ppo import ActorCritic


class AdaptiveShopTests(unittest.TestCase):
    def test_mixture_is_marginal_not_conditional_on_source(self):
        self.assertAlmostEqual(mixture_probability(.10, 4), .13)
        self.assertAlmostEqual(mixture_probability(0, 2), .10)

    def test_promotion_requires_validation_and_superiority(self):
        baseline = {i: (10, False, False) for i in range(64)}
        better = {i: (11, False, False) for i in baseline}
        worse = {i: (9, False, False) for i in baseline}
        self.assertTrue(decision(baseline, better, baseline, better, minimum=32)['promoted'])
        self.assertFalse(decision(baseline, better, baseline, worse, minimum=32)['promoted'])
        self.assertFalse(decision(baseline, baseline, baseline, baseline, minimum=32)['promoted'])

    def test_guidance_edits_only_reward_head_and_keeps_mask(self):
        model = ActorCritic([128,128])
        obs = np.ones((16,583), np.float32)
        mask = np.tile([True, True, False, True, True], (16,1))
        before = {k: v.detach().clone() for k,v in model.state_dict().items()}
        scores = np.tile([5.,0.,10.,0.,0.], (16,1))
        metrics = guide(model, obs, mask, scores, strength=.15)
        self.assertGreater(metrics['gold_probability_change'], 0)
        self.assertLess(metrics['kl'], .01)
        for key,value in model.state_dict().items():
            if not key.startswith('reward_head.'):
                self.assertTrue(torch.equal(value,before[key]), key)

    def test_long_schedule_has_final_frozen_shop(self):
        self.assertEqual(mode(0,25000,1000000),'uniform')
        self.assertEqual(mode(75000,25000,1000000),'paired')
        self.assertEqual(mode(100000,25000,1000000),'adaptive')
        self.assertEqual(mode(1000000,25000,1000000),'adaptive')

    def test_promotion_can_distinguish_floor_100_clear_from_defeat(self):
        lost = {i:(100,False,False) for i in range(32)}
        cleared = {i:(100,True,False) for i in lost}
        result = decision(lost, cleared, lost, cleared, minimum=32)
        self.assertTrue(result['promoted'])
        self.assertEqual(result['development']['mean_floor_change'], 0)

    def test_gold_edit_reaches_later_heads_and_is_reproducible(self):
        run = Run(17,42,10000,mode='paired')
        run.game.phase = 'shop'
        run.game.gold = 1500
        run.state = run.game.get_state()
        key = next(k for k in run.game.player.learned_skills if k!='lock' and run.game.player.skill_costs[k]>=3)
        original = run.game.player.skill_costs[key]
        calls = []
        features = np.zeros(1370,dtype=np.float32)

        def choices(game,head,rng,count,area,route):
            calls.append((head,game.player.skill_costs[key]))
            options = [Candidate((),features)]
            if head==0:
                options.append(Candidate((('shop_menu','costs'),('reduce_skill_cost',key),('shop_back',)),features))
            return options

        policy = {},(False,True,True),True
        with patch('adaptive_shop.gameplay.configurations',side_effect=choices), patch(
                'adaptive_shop.gameplay.scores',return_value=np.array([0.])):
            first = planning(run,policy,2,override=0)[0]
            second = planning(run,policy,2,override=0)[0]
        self.assertEqual(first,second)
        self.assertEqual(calls[:3],[(0,original),(1,original-1),(2,original-1)])
        self.assertEqual(run.game.player.skill_costs[key],original)


if __name__ == '__main__':
    unittest.main()
