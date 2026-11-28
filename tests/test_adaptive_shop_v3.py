from io import BytesIO
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from adaptive_shop.gameplay import Run
from adaptive_shop.reward import guide
from adaptive_shop_v3.planning import Choice, local_candidate, proposals, choose, snapshot, treatment
from adaptive_shop_v3.promotion import assess, choose_promotion
from adaptive_shop_v3.runner import DEFAULT, Experiment, configuration
from baseline_shop.configurations import replay
from baseline_shop.model import ConfigurationValue
from combat_rl.ppo import ActorCritic


class V3Tests(unittest.TestCase):
    def shop(self):
        run = Run(17,42,10000,mode='paired')
        run.game.phase = 'shop'
        run.game.floor = run.floor = 10
        run.game.gold = 1800
        run.state = run.game.get_state()
        return run

    def test_local_search_replays_and_preserves_original(self):
        run = self.shop()
        original = run.game.get_state()
        for head in range(3):
            paths = [local_candidate(run.game,head,random.Random(i)) for i in range(10)]
            self.assertTrue(any(paths))
            for actions in paths:
                replay(run.game,actions)
        self.assertEqual(run.game.get_state(),original)

    def test_frozen_encoder_and_candidate_budget(self):
        run = self.shop()
        model = ConfigurationValue()
        saved = snapshot(model,(True,True,True),25000,8,(None,)*3)
        for head,policy in enumerate(saved):
            options = proposals(run.game,head,run.policy_seed,policy,run.context())
            self.assertLessEqual(len(options),8)
        before = choose(run.game,1,run.policy_seed,saved[1],run.context())
        with torch.no_grad():
            model.encoder[0].weight.fill_(42.)
        self.assertEqual(choose(run.game,1,run.policy_seed,saved[1],run.context()),before)
        payload = BytesIO()
        torch.save(saved,payload)
        payload.seek(0)
        restored = torch.load(payload,weights_only=False)
        self.assertEqual(choose(run.game,1,run.policy_seed,restored[1],run.context()),before)

    def test_negative_combination_is_not_promoted(self):
        base = {i:(10,False,False) for i in range(32)}
        gain = {i:(11,False,False) for i in base}
        drop = {i:(9,False,False) for i in base}
        ok, good = assess(base,gain,base,gain,minimum=32,step=1)
        bad, rejected = assess(base,gain,base,drop,minimum=32,step=2)
        self.assertTrue(ok)
        self.assertFalse(bad)
        self.assertEqual(choose_promotion({'gold':good,'gold+stats':rejected}),'gold')

    def test_transfer_protects_combat_and_masks_with_v3_limits(self):
        model = ActorCritic([128,128])
        prior = {k:v.clone() for k,v in model.state_dict().items()}
        x = np.ones((16,583),np.float32)
        mask = np.tile([True, True, False, True, True],(16,1))
        values = np.tile([6.,0.,100.,0.,0.],(16,1))
        r = guide(model,x,mask,values,.50,updates=40,kl_limit=.05,temperature=2.)
        self.assertGreater(r['gold_probability_change'],0.)
        self.assertEqual(r['probability_changes'][2],0.)
        self.assertLessEqual(r['kl'],.05)
        for k,v in model.state_dict().items():
            if not k.startswith('reward_head.'):
                self.assertTrue(torch.equal(v,prior[k]),k)

    def test_local_treatment_starts_at_incumbent_state(self):
        run = self.shop()
        anchor = (('shop_menu','stats'),('adjust_stat','attack',-1),('shop_back',))
        before = run.game.player.base_attack
        with patch('adaptive_shop_v3.planning.choose',return_value=Choice(anchor)), patch(
                'adaptive_shop_v3.planning.origin_rng',return_value=random.Random(1)), patch(
                'adaptive_shop_v3.planning.local_candidate',return_value=()) as local:
            choice = treatment(run.game,1,run.policy_seed,None,8,run.context())
        self.assertEqual(choice.source,'local')
        self.assertEqual(choice.actions,anchor)
        self.assertEqual(local.call_args.args[0].player.base_attack,before-1)
        self.assertEqual(run.game.player.base_attack,before)

    def test_promotion_replaces_one_head_without_replacing_a_regressing_head(self):
        experiment = object.__new__(Experiment)
        model = ConfigurationValue()
        original = snapshot(model,(True,True,True),50_000,8,(None,)*3)
        experiment.baseline = original
        experiment.shop = model
        experiment.state = {'trained':[2,2,2],'baseline_version':1}
        experiment.cfg = {'candidates':8,'promotion_minimum':32}
        experiment.publish = Mock()
        base = {i:(10,False,False) for i in range(32)}

        def panel(pool,step,name,policy,split=None):
            score = {'adaptive':10,'variant_combined':12,'variant_gold':11,
                     'variant_stats':9,'variant_skills':10}[policy]
            return {i:(score,False,False) for i in base}

        experiment.panel = panel
        result = experiment.promotion(None,100_000,base)
        self.assertEqual(result['selected'],'gold')
        self.assertEqual(experiment.baseline[0].step,100_000)
        self.assertIs(experiment.baseline[1],original[1])
        self.assertIs(experiment.baseline[2],original[2])

    def test_disk_guard_saves_before_shutdown_and_production_has_no_early_stop(self):
        experiment = object.__new__(Experiment)
        experiment.boundaries_on_disk = Mock(return_value={'free_bytes':10,'reserve_bytes':20})
        experiment.save = Mock()
        with self.assertRaises(OSError):
            experiment.check_disk(pool='worker')
        experiment.save.assert_called_once_with('worker')
        cfg = configuration(DEFAULT)
        self.assertEqual(cfg['first_stop'],1_025_000)
        self.assertEqual(cfg['episodes']-cfg['alternating_episodes'],25_000)


if __name__=='__main__':
    unittest.main()
