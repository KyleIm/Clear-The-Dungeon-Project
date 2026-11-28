"""Alternating PPO with a promoted Shop incumbent and Reward-only experience transfer."""

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from baseline_shop.runner import (Experiment as V1, INITIAL, ROOT, configuration as base_config,
                                  require_isolated_output)
from baseline_shop.configurations import HEADS
from combat_rl.performance import Profile, Resources
from combat_rl.runner import load as load_ppo, seed
from combat_rl.workers import Workers
from experiments.random_baseline_large import run_lock

from .promotion import decision, superiority
from .storage import Store
from .workers import handle

VERSION = 'adaptive-shop-v2'
DEFAULT = ROOT / 'adaptive_shop/training.json'
PLAN_VERSION = 'v2-config-1'


def configuration(path):
    cfg = base_config(path)
    span = cfg['segment_episodes']
    if (not 0 < cfg['first_stop'] <= cfg['alternating_episodes']
            or cfg['first_stop'] % span or cfg['alternating_episodes'] % (2*span)
            or cfg['episodes'] != cfg['alternating_episodes'] + span):
        raise ValueError('Expected whole alternating cycles and one final PPO segment')
    for k in ('heldout_episodes','promotion_minimum','reward_rows','reward_epochs','reward_support','probe_episodes'):
        if type(cfg[k]) is not int or cfg[k]<1:
            raise ValueError(f'{k} must be positive')
    return cfg


def mode(completed, span, alternating):
    block = completed // span
    if completed >= alternating:
        return 'adaptive'
    return 'uniform' if block==0 else 'paired' if block%2 else 'adaptive'


def arrays(model):
    return {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}


class Experiment(V1):
    def __init__(self,cfg,initial,old,store,directory,resources):
        from .reward import RewardValue

        super().__init__(cfg,initial,old,store,directory,resources)
        self.state.setdefault('baseline_version',0)
        self.state.setdefault('candidate_segment',0)
        self.baseline = (old['baseline'] if old else (None,(False,False,False),True))
        with self.torch.random.fork_rng(devices=[]):
            self.reward_q = RewardValue().to(self.ppo.device)
        if old:
            self.reward_q.load_state_dict(old['reward_q'])
            if old['policy_manifest'] != self.manifest():
                raise ValueError('Checkpoint configuration generator is incompatible')
        self.shop_actor = self.shop

    def manifest(self):
        return dict(generator=PLAN_VERSION,candidates=self.cfg['candidates'],
                    choice='argmax P(win)-P(loss), unchanged value 0, stable first tie',
                    rng='Python Random(v2-config-1:episode_policy_seed:floor:head)')

    def pool(self):
        return Workers(self.cfg['processes'],self.cfg['workers']//self.cfg['processes'],
                       self.profile,apply_handler=handle)

    def save(self,pool,pre=False,archive=False):
        self.state['wall_seconds'] = self.prior_wall+perf_counter()-self.started
        workers = pool.call('snapshot',dict.fromkeys(self.active))
        with self.profile.measure('checkpoint'):
            payload = dict(version=VERSION, run_id=self.run_id, config=self.cfg, ppo_config=self.ppo_config,
                           state=self.state,active=self.active,workers=workers,marks=self.store.marks(),
                           model=self.ppo.state_dict(),ppo_opt=self.ppo_opt.state_dict(),
                           shop=self.shop.state_dict(),shop_opt=self.shop_opt.state_dict(),baseline=self.baseline,
                           reward_q=self.reward_q.state_dict(),policy_manifest=self.manifest(),
                           rng=self.rng.getstate(),torch_rng=self.torch.get_rng_state(),
                           cuda_rng=self.torch.cuda.get_rng_state_all() if self.cfg['device']=='cuda' else None,
                           profile_totals=dict(self.profile.totals),profile_counts=dict(self.profile.counts))
            tmp = self.directory/'latest.tmp'
            latest = self.directory/'latest.pt'
            self.torch.save(payload,tmp)
            tmp.replace(latest)
            if pre or archive:
                count = self.state['episodes']
                suffix = '-pre' if pre else ''
                copyfile(latest,tmp)
                tmp.replace(self.directory/f'episode-{count:07d}{suffix}.pt')

    def publish(self,pool,candidate=False,ppo=False):
        weights = arrays(self.actor) if ppo else None
        models = dict(adaptive=(weights,self.baseline))
        if candidate:
            models['candidate'] = weights,(arrays(self.shop),tuple(self.state['trained']),
                                           self.state['baseline_version']==0)
        pool.call('policy',dict.fromkeys(range(self.cfg['processes']),models))

    def request(self,split,number,mode,step=0):
        if split=='heldout' or split=='probe':
            master = self.cfg['evaluation_seed']
            args = (seed(master,f'v2-{split}-game-{step}',number),seed(master,f'v2-{split}-policy-{step}',number),
                    self.cfg['action_cap'])
        else:
            args,_ = super().request(split,number,mode)
        return args,dict(mode=mode,version=self.state['version'],shop_version=self.state['baseline_version'])

    def step(self,pool,active,mode,deterministic):
        choices,logs,values,probs = self.choose(active,deterministic)
        requests = {slot:(int(action),prob,self.state['version'])
                    for slot,action,prob in zip(active,choices,probs)}
        replies = pool.call('step',requests)
        shops = [slot for slot,r in replies.items() if r.phase=='shop']
        labels = []
        if shops:
            with self.profile.measure('baseline_comparison' if mode=='paired' else 'shop_execution'):
                edited = pool.call('shop',dict.fromkeys(shops,(mode,self.cfg)))
            for slot,(result,examples) in edited.items():
                prior = replies[slot]
                replies[slot] = result._replace(reward=result.reward+prior.reward,actions=result.actions+prior.actions)
                labels.extend(examples)
        return replies,labels,choices,logs,values

    def panel(self,pool,step,name,policy,split='evaluation'):
        count = self.cfg['eval_episodes' if split=='evaluation' else
                         'heldout_episodes' if split=='heldout' else 'probe_episodes']
        active, rows = {}, {}
        started = actions = 0
        begin = perf_counter()

        def refill(slots):
            nonlocal started
            requests = {}
            for slot in slots:
                if started>=count:
                    break
                requests[slot] = self.request(split,started,policy,step)
                started += 1
            active.update({slot: reply.observation for slot,reply in pool.call('reset',requests).items()})

        refill(range(self.cfg['workers']))
        while active:
            replies,*_ = self.step(pool,active,policy,True)
            finished = []
            for slot,reply in replies.items():
                actions += reply.actions
                row = reply.result
                if row is None:
                    active[slot] = reply.observation
                    continue
                rows[row['game_seed'],row['policy_seed']] = (row['floor'],row['outcome']=='cleared',row['outcome']=='timeout')
                self.store.episode('eval_'+name,step,dict(row,segment=step))
                del active[slot]
                finished.append(slot)
            refill(finished)
        self.store.evaluation(step,name,dict(episodes=count,actions=actions,seconds=perf_counter()-begin))
        print(f'Evaluation {step:,} | {name} | {count} games | {actions:,} actions',flush=True)
        return rows

    def transfer(self,step,old_version,promoted):
        import numpy as np
        from combat_rl.environment import REWARD_CHOICES
        from .reward import RewardValue, fit, guide

        started = perf_counter()
        take = partial(self.store.reward_batch,step,old_version)
        training = take(False,self.cfg['reward_rows'],self.rng)
        heldout = take(True,self.cfg['reward_rows']//4,self.rng)
        forced = [sum(row[4]=='random' and row[2]==i for row in training) for i in range(5)]
        result = dict(training_rows=len(training),heldout_rows=len(heldout),
                      forced_training_counts=dict(zip(REWARD_CHOICES,forced)),shop_version=old_version,
                      policy_shift=promoted,applied=False)
        if not training or not heldout:
            return result
        self.reward_q = RewardValue().to(self.ppo.device)
        device = self.ppo.device
        torch = self.torch
        rng = np.random.default_rng(self.rng.randrange(2**63))
        fit_start = perf_counter()
        result['q_loss'] = fit(self.reward_q,training,device,rng,self.cfg['reward_epochs'])
        result['q_seconds'] = perf_counter()-fit_start
        x = np.stack([row[0] for row in heldout])
        legal = np.stack([row[1] for row in heldout])
        with torch.no_grad():
            q = self.reward_q(torch.as_tensor(x,device=device)).cpu().numpy()
            observed = np.asarray([row[3] for row in heldout])
            chosen = np.asarray([row[2] for row in heldout])
            result['heldout_rmse_floor'] = float(np.sqrt(np.mean((q[np.arange(len(x)),chosen]-observed)**2)))
            result['q_distribution'] = {name:dict(n=int(legal[:,i].sum()),
                quantiles=np.quantile(q[legal[:,i],i],(.1,.5,.9)).tolist())
                for i,name in enumerate(REWARD_CHOICES) if legal[:,i].any()}
        support = np.asarray(forced)>=self.cfg['reward_support']
        guide_start = perf_counter()
        outcome = guide(self.ppo,x,legal,q,strength=.025 if promoted else .10,supported=support)
        result.update(outcome,aux_seconds=perf_counter()-guide_start,total_seconds=perf_counter()-started)
        if outcome['applied']:
            for p in self.ppo.reward_head.parameters():
                self.ppo_opt.state.pop(p,None)
            if self.actor is not self.ppo:
                self.actor.load_state_dict(self.ppo.state_dict())
            self.state['version'] += 1
        return result

    def boundary(self,pool,step,adapt):
        profile = self.profile
        try:
            self.profile = pool.profile = Profile()
            self.publish(pool,candidate=adapt,ppo=True)
            info = dict(baseline_before=self.state['baseline_version'])
            if step <= self.cfg['segment_episodes']:
                self.panel(pool,step,'uniform','uniform')
            self.panel(pool,step,'baseline','baseline')
            incumbent = self.panel(pool,step,'adaptive','adaptive')
            if adapt:
                start = perf_counter()
                candidate = self.panel(pool,step,'candidate','candidate')
                validated = self.panel(pool,step,'heldout_adaptive','adaptive','heldout')
                validation = self.panel(pool,step,'heldout_candidate','candidate','heldout')
                result = decision(incumbent,candidate,validated,validation,minimum=self.cfg['promotion_minimum'],seed=step)
                result['promotion_seconds'] = perf_counter()-start
                for head,name in enumerate(HEADS):
                    independent = self.panel(pool,step,name,f'candidate_{head}')
                    result[name] = superiority(incumbent,independent,minimum=self.cfg['promotion_minimum'],
                                               alpha=.0025,seed=step+head)
                if result['promoted']:
                    self.baseline = (arrays(self.shop),tuple(self.state['trained']),
                                     self.state['baseline_version']==0)
                    self.state['baseline_version'] += 1
                info['promotion'] = result
                self.publish(pool,ppo=True)
                self.panel(pool,step,'accepted_pre','adaptive')
                info['transfer'] = self.transfer(step,info['baseline_before'],result['promoted'])
                self.publish(pool,ppo=True)
                self.panel(pool,step,'accepted_post','adaptive')
            self.publish(pool,candidate=adapt or any(self.state['trained']),ppo=True)
            self.panel(pool,step,'resource_probes','probe','probe')
            info.update(baseline_after=self.state['baseline_version'],profile=self.profile.summary(),
                        resources=self.resources.summary())
            self.store.boundary(step,info)
        finally:
            self.profile = pool.profile = profile

    def run(self,stop):
        span = self.cfg['segment_episodes']
        limit = self.cfg['alternating_episodes']
        with self.pool() as pool:
            if self.old:
                pool.call('restore',self.old['workers'])
            published = None
            while True:
                count = self.state['episodes']
                if count%span==0 and count!=self.state['last_eval']:
                    assert not self.active
                    adapt = 0 < count <= limit and count%(2*span)==0
                    self.save(pool,pre=adapt)
                    self.boundary(pool,count,adapt)
                    self.state['last_eval']=count
                    self.save(pool,archive=True)
                    published = None
                if count>=stop:
                    return
                kind = mode(count,span,limit)
                end = (count//span+1)*span
                if kind=='paired' and self.state['candidate_segment']!=end:
                    from baseline_shop.model import ConfigurationValue
                    self.shop = self.shop_actor = ConfigurationValue().to(self.ppo.device)
                    self.shop_opt = self.torch.optim.Adam(self.shop.parameters(),lr=self.cfg['shop_lr'])
                    self.state.update(candidate_segment=end,trained=[0,0,0])
                stamp = (self.state['baseline_version'],self.state['version'] if kind=='paired' else None)
                if published!=stamp:
                    self.publish(pool,ppo=kind=='paired')
                    published = stamp
                self.refill(pool,(slot for slot in range(self.cfg['workers']) if slot not in self.active),end,kind)
                self.rollout(pool,end,kind)
                if self.state['rounds']%self.cfg['checkpoint_rounds']==0:
                    self.save(pool)


def train(cfg,directory,initial_path,*,resume=False,stop=None):
    import torch

    require_isolated_output(directory,initial_path)
    previous_v1 = (ROOT/'data/baseline_relative_shop').resolve()
    if directory.resolve().is_relative_to(previous_v1):
        raise ValueError('Use an output directory outside the preliminary V1 experiment')
    if stop is None:
        stop = cfg['episodes'] if resume else cfg['first_stop']
    if not 0<stop<=cfg['episodes'] or stop%cfg['segment_episodes']:
        raise ValueError('Choose a positive segment boundary within the budget')
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'latest.pt'
    database = directory/'training.sqlite3'
    with run_lock(directory):
        if resume:
            if not path.is_file() or not database.is_file():
                raise ValueError('Resume requires latest.pt and its SQLite database')
            old = torch.load(path,map_location='cpu',weights_only=False)
            if old.get('version')!=VERSION or old['config']!=cfg:
                raise ValueError('Resume requires matching V2 checkpoint and settings')
            if old['state']['started']>stop:
                raise ValueError('Resume target precedes existing started episodes')
            initial = None
        else:
            if path.exists() or database.exists():
                raise ValueError('Choose a fresh directory or resume the existing run')
            old,initial = None,load_ppo(initial_path)
            if initial['state']['episodes']!=1_000_000 or initial['config']['hidden_dims']!=[128,128]:
                raise ValueError('Use the final Phase 3B-2 Combat and Reward PPO checkpoint')
        with Store(database,batch_size=1024) as store, Resources() as resources:
            Experiment(cfg,initial,old,store,directory,resources).run(stop)


def main():
    from .report import report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','resume','report'))
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--initial',type=Path,default=INITIAL)
    parser.add_argument('--stop-at',type=int)
    args = parser.parse_args()
    cfg = configuration(args.config)
    directory = args.output or ROOT/cfg['output']
    if args.command!='report':
        train(cfg,directory,args.initial,resume=args.command=='resume',stop=args.stop_at)
    print(report(directory))


if __name__=='__main__':
    main()
