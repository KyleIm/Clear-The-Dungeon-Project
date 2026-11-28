"""Run the V3 long experiment and keep its checkpoints separate from earlier phases."""

import argparse
from pathlib import Path
from shutil import disk_usage
from time import perf_counter

from adaptive_shop.runner import Experiment as V2, INITIAL, ROOT, arrays, train as train_common
from baseline_shop.configurations import HEADS
from baseline_shop.runner import configuration as common_config, require_isolated_output
from combat_rl.performance import Profile
from combat_rl.workers import Workers

from .planning import GENERATOR, snapshot
from .promotion import assess, choose_promotion
from .workers import handle

VERSION = 'adaptive-shop-v3'
DEFAULT = ROOT / 'adaptive_shop_v3/training.json'
GIB = 1024**3


def configuration(path):
    cfg = common_config(path)
    span = cfg['segment_episodes']
    if (cfg['episodes']!=cfg['alternating_episodes']+span or cfg['alternating_episodes']<=0
            or cfg['alternating_episodes']%(2*span) or cfg['first_stop']!=cfg['episodes']
            or not 2<=cfg['candidates']<=8):
        raise ValueError('V3 needs whole alternating cycles, a final PPO segment and at most eight candidates')
    for key in ('heldout_episodes','promotion_minimum','reward_rows','reward_epochs','reward_support',
                'probe_episodes','disk_reserve_mib'):
        if type(cfg[key]) is not int or cfg[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['promotion_minimum']>min(cfg['eval_episodes'],cfg['heldout_episodes']):
        raise ValueError('The evaluation panels must support the minimum pair count')
    return cfg


def free_bytes(path):
    return disk_usage(path).free


class Experiment(V2):
    checkpoint_version = VERSION

    def __init__(self,cfg,initial,old,store,directory,resources):
        super().__init__(cfg,initial,old,store,directory,resources)
        if old is None:
            self.baseline = (None,)*3
        self.variants = {}

    def manifest(self):
        return dict(generator=GENERATOR,candidates=self.cfg['candidates'],
                    selection='frozen head encoder and recursive incumbent proposal anchor',
                    rng='v3-mixed-1:episode_policy_seed:floor:head:stream:boundary',
                    treatment_probability={'global':.5,'local':.5},tie='keep anchor')

    def guidance_settings(self,promoted):
        return dict(strength=.10 if promoted else .50, updates=40, kl_limit=.05, temperature=2.)

    def q_diagnostics(self,rows,legal,q):
        from collections import defaultdict
        import numpy as np
        from combat_rl.environment import REWARD_CHOICES

        result = []
        groups = defaultdict(list)
        for index,row in enumerate(rows):
            data = row[5]
            groups['hp',data['hp_bucket']].append(index)
            groups['shop_distance',str(data.get('shop_distance','unavailable'))].append(index)
            drift = [data['current_stats'][k]-data['base_stats'][k] for k in data['current_stats']]
            direction = ('mixed' if min(drift)<0<max(drift) else 'below' if min(drift)<0 else
                         'above' if max(drift)>0 else 'same')
            groups['stat_drift',direction].append(index)
        for (axis,value),indices in sorted(groups.items()):
            for i,name in enumerate(REWARD_CHOICES):
                matched = [index for index in indices if legal[index,i]]
                if matched:
                    result.append(dict(axis=axis,bucket=value,reward=name,states=len(matched),
                                       quantiles=np.quantile(q[matched,i],(.1,.5,.9)).tolist()))
        return dict(q_by_context=result)

    def pool(self):
        return Workers(self.cfg['processes'],self.cfg['workers']//self.cfg['processes'],
                       self.profile,apply_handler=handle)

    def publish(self,pool,candidate=False,ppo=False):
        models = dict(adaptive=self.baseline,variants=self.variants,
                      ppo=arrays(self.actor) if ppo else None)
        pool.call('policy',dict.fromkeys(range(self.cfg['processes']),models))

    def boundaries_on_disk(self):
        files = ('training.sqlite3','training.sqlite3-wal')
        return dict(free_bytes=free_bytes(self.directory),
                    database_bytes=sum(p.stat().st_size for name in files
                                       if (p:=self.directory/name).exists()),
                    reserve_bytes=self.cfg['disk_reserve_mib']*1024**2)

    def check_disk(self,pool=None):
        space = self.boundaries_on_disk()
        if space['free_bytes'] < space['reserve_bytes']:
            if pool is not None:
                self.save(pool)
            raise OSError(f"Free space fell below the configured reserve of {space['reserve_bytes']/1024**2:.0f} MiB. "
                          'Resume after making space on the output drive.')

    def run(self,stop):
        space = self.boundaries_on_disk()
        print(f"Output drive free: {space['free_bytes']/GIB:.1f} GiB | "
              f"database + WAL: {space['database_bytes']/GIB:.2f} GiB",flush=True)
        self.check_disk()
        super().run(stop)

    def rollout(self,pool,boundary,mode):
        super().rollout(pool,boundary,mode)
        self.check_disk(pool)

    def promotion(self,pool,step,incumbent):
        started = perf_counter()
        original = self.baseline
        candidate = snapshot(self.shop,self.state['trained'],step,self.cfg['candidates'],original)
        self.variants = {'variant_combined':candidate}
        for i,head in enumerate(HEADS):
            self.variants['variant_'+head] = tuple(candidate[j] if i==j else original[j] for j in range(3))
        self.publish(pool)
        validation_base = self.panel(pool,step,'heldout_adaptive','adaptive','heldout')
        results = {}
        edits = {}

        def measure(key,replaced):
            policy = 'variant_'+key
            edits[key] = replaced
            name = 'candidate' if key=='combined' else key
            development = self.panel(pool,step,name,policy)
            validation = self.panel(pool,step,'heldout_'+name,policy,'heldout')
            _, result = assess(incumbent,development,validation_base,validation,
                               minimum=self.cfg['promotion_minimum'],step=step+len(results))
            result['replaced_heads'] = [HEADS[i] for i in replaced]
            results[key] = result

        changed = tuple(i for i in range(3) if candidate[i] is not original[i])
        measure('combined',changed)
        for i, head in enumerate(HEADS):
            measure(head,(i,) if i in changed else ())
            if i not in changed:
                results[head].update(promoted=False,reason='insufficient contrasting training examples')
        measured_regressions = [head for head in HEADS if min(results[head]['development']['mean_utility_change'],
                                                              results[head]['heldout']['mean_utility_change'])<0]
        if measured_regressions:
            results['combined'].update(promoted=False,
                reason='includes a head with measured regression: '+', '.join(measured_regressions))
        winners = tuple(i for i,h in enumerate(HEADS) if results[h]['promoted'])
        if 1<len(winners)<len(changed):
            self.variants['variant_subset'] = tuple(candidate[i] if i in winners else original[i] for i in range(3))
            self.publish(pool)
            measure('subset',winners)
        if not changed:
            results['combined'].update(promoted=False,reason='no trained candidate heads')
        selected = choose_promotion(results)
        adopted = edits[selected] if selected else ()
        if selected:
            self.baseline = self.variants['variant_'+selected]
            self.state['baseline_version'] += 1
        for name,row in results.items():
            row['selected'] = name==selected
            if row['promoted'] and name!=selected:
                row['reason'] = 'eligible; another policy had stronger development evidence'
        highest = results[selected or 'combined']
        changes = {head:dict(before=original[i].step if original[i] else 0,
                             after=self.baseline[i].step if self.baseline[i] else 0,
                             adopted=i in adopted,
                             reason=('validated replacement in '+selected if i in adopted else
                                     results[head]['reason'] if not results[head]['promoted'] else
                                     'retained in the selected stack' if selected else 'incumbent retained'))
                   for i,head in enumerate(HEADS)}
        return dict(promoted=selected is not None, selected=selected, head_changes=changes, attempts=results,
                    development=highest['development'],heldout=highest['heldout'],
                    promotion_seconds=perf_counter()-started)

    def boundary(self,pool,step,adapt):
        profile = self.profile
        try:
            self.profile = pool.profile = Profile()
            self.variants = {}
            self.publish(pool)
            info = dict(baseline_before=self.state['baseline_version'])
            if step<=self.cfg['segment_episodes']:
                self.panel(pool,step,'uniform','uniform')
            self.panel(pool,step,'baseline','baseline')
            incumbent = self.panel(pool,step,'adaptive','adaptive')
            if adapt:
                info['promotion'] = self.promotion(pool,step,incumbent)
                self.publish(pool)
                self.panel(pool,step,'accepted_pre','adaptive')
                info['transfer'] = self.transfer(step,info['baseline_before'],info['promotion']['promoted'])
                self.publish(pool)
                self.panel(pool,step,'accepted_post','adaptive')
            self.publish(pool,ppo=True)
            self.panel(pool,step,'resource_probes','probe','probe')
            info.update(baseline_after=self.state['baseline_version'],profile=self.profile.summary(),
                        resources=self.resources.summary(),disk=self.boundaries_on_disk(),
                        wall_seconds=self.prior_wall+perf_counter()-self.started)
            self.store.boundary(step,info)
        finally:
            self.profile = pool.profile = profile


def train(cfg,directory,initial_path,*,resume=False,stop=None):
    require_isolated_output(directory,initial_path)
    for old_name in ('adaptive_shop_v2','baseline_relative_shop'):
        if directory.resolve().is_relative_to((ROOT/'data'/old_name).resolve()):
            raise ValueError('Use a separate directory outside preliminary Shop experiments')
    directory.mkdir(parents=True,exist_ok=True)
    if free_bytes(directory) < cfg['disk_reserve_mib']*1024**2:
        raise OSError('Output drive has less free space than the configured reserve')
    return train_common(cfg,directory,initial_path,resume=resume,stop=stop,experiment_class=Experiment)


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
