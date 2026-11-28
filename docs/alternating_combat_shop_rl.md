# Phase 3C revision · Alternating Masked PPO and Masked DQN

The existing launcher now starts the revised experiment in `data/alternating_ppo_dqn`. It initializes Combat and Reward from the verified Phase 3B-2 million-episode PPO checkpoint and initializes Shop DQN independently. Earlier PPO and shop-candidate checkpoints and databases remain in their original directories. Candidate checkpoints are incompatible with the new schema.

## Alternation and evaluation

The schedule adds 1,000,000 games in 40 segments of exactly 25,000. Odd segments update the original 583-input, 128/128 shared Combat/Reward PPO and hold the Shop model fixed. Even segments update Shop DQN and hold PPO fixed. The first segment and its boundary evaluations use the original uniform-random shop behavior. The first Shop segment may warm up from verified compatible observations of those new random games. Every phase retains the original +1-per-new-floor PPO reward. The floor-35 branch is still uniform over the two original choices.

No game straddles a segment boundary. The initial PPO checkpoint is evaluated at step zero, and each following boundary uses the same separate 128 fixed game/policy seed pairs. Combat and Reward evaluate with masked argmax. Shop evaluation uses greedy DQN after the bootstrap segment. The engine, chosen and effective Ella route, and historical reference results remain separate. [Phase 3B-2 baseline](combat_rl_1m/training_report.md) · [Combat and Reward PPO settings](combat_reward_rl.md)

## Ordered shop actions

Candidate generation required a fixed number of short, randomly edited exits and provided a score only for the completed configuration. The revised action interface exposes every legal edit in the active stage. It can respond to a realized reroll, evaluate much longer reallocations, and give individual edits their own learning targets.

| Stage | Output IDs | Mapping | Modification limit |
|---|---|---|---:|
| Gold Expenditure | 0–97 | 0–87 reduce the cost of each registry skill. 88–92 buy blue key, red key and the three ordered mana potions. 93–95 change type to Undead, Elemental or Beast. 96 rerolls additional stats. 97 finishes. | 25 |
| Stat Redistribution | 0–17 | For each of ATK, DEF, HP, SPD in order, expose −10, −1, +1, +10. 16 resets redistribution. 17 finishes. | 100 |
| Skill Configuration | 0–540 | 0–527 replace a slot's skill, grouped by slot 1–6 and then registry skill. 528–533 increase uses by slot, 534–539 decrease uses, and 540 finishes. | 125 |

All registry IDs follow ascending source skill ID order. Each head ends voluntarily or on reaching its modification limit and progression is forward only. Menu entries and returns use real simulator actions without consuming the modification budget. They do count toward the existing episode action cap, as do all executed primitive actions. Finishing the final head leaves the shop. An action-cap interruption cannot create a fake shop exit. Underspending, unallocated points, and zero-use equipped skills remain legal. There is no repetition penalty.

Masks come from the original simulator's menu-specific legal actions, including prerequisites and prices, with a guaranteed Finish action. The stat head retains the source's changing base values until the original Confirm/Back action refreshes current stats and HP. In the original Ren’Py snapshot, redistribution edits are at `script.rpy:1582–1660` and the refresh is at `4332–4335`. The Reroll body deducts the price, redistributes the same total through two random cuts, synchronizes the additional current stats and doubles the price (`1661–1684`). The source audit's isolated compilation issue for that function remains documented. This adapter calls the existing simulator Reroll and reads resulting stats, gold, and price after the action. Steam runtime behavior was outside this validation. [Simulator fidelity audit](simulator_fidelity.md)

## Network inputs

The shared input has **688 FP32 nodes**. The original 671 positions now describe the current pre-action shop state.

| Feature group | Nodes | Encoding |
|---|---:|---|
| Player | 26 | Current and maximum HP, current and base ATK/DEF/SPD, current ACC/EVA/DR, four current type bits and 11 status-presence bits. |
| Six equipped slots | 450 | Occupancy, use count, four type bits, power, accuracy, priority, 65 category bits, current cost. Locked slots are zero. Depleted occupied slots retain their skill descriptors. |
| Acquired skills and current costs | 176 | Two 88-wide vectors in ascending skill ID order. The cost position stays zero when unlearned. The registry retains the dummy Strike and non-learnable Sacrifice positions. |
| Maximum, used and remaining mana | 3 | Current budget and allotments. Used mana is allotted uses multiplied by actual skill cost. |
| Unused points, gold, floor | 3 | Unspent points and current gold scaled, floor divided by 100. |
| Region and event choice | 13 | Eleven registry-order region bits and two branch bits. Ordinary shops use the preceding region; floor 99 keeps its area. A/B are the recorded choices, and both are zero before the event. |
| Active shop head and remaining budget | 4 | Three one-hot head bits and the fraction of the current modification allowance left. |
| Reroll price and purchase flags | 6 | Current reroll price plus flags for both keys and each of the three mana potions. This captures purchase legality even when slot or mana counts alone cannot distinguish prior purchases. |
| Base additional stats and base type | 7 | Base ACC/EVA/DR and four base type bits, separate from the visible current values. |
| **Total** | **688** | |

The type order is Normal, Beast, Elemental, Undead. Status bits are stun, poison, sprout, regeneration, target, sleep, reverse, aurora, immunity, imitation and helmet. Category bits use ascending IDs from the 65 registered categories. Locked slots are already encoded in the occupancy bits. Base HP is inferable from maximum HP divided by two (`script.rpy:1517–1518`). Redistribution uses the base main stats and unused points already present in the input. Signed `log1p(abs(x)/unit)` uses unit 100 for actor stats, mana, points and reroll price, unit 10 for use counts and costs, and unit 1,000 for gold. Power and accuracy are divided by 100.

## Q targets and replay

The Shop network uses shared 128/128 tanh layers and three linear output heads. For an internal transition, including a head change, the online network selects the next legal argmax and the target network evaluates it. The Bellman target is `r + Q_target(next_state, argmax_legal Q_online)`, with discount 1. Internal rewards are zero. The final transition is held pending across the subsequent game segment. It receives 1 on the first success milestone and 0 on earlier defeat. A timeout without a resolved milestone is censored and supplies no terminal target. Internal transitions already observed before censoring remain available. Raw Q values are unconstrained during learning; reliability bins clip them to [0,1] for display.

| Shop | Success milestone |
|---|---|
| 10–80 | Entry to the next regular shop following that stage boss. |
| 90 | Entry to floor 100 after the required floor-99 continuation. |
| 99 | Final boss defeated and the game reaches its cleared state. |

The source assigns Pickpocket/Help at floor 35 (`3823–3870`). At floor 99, the initial Gate encounter is followed by another Ella battle on effective A, or a special shop on effective B. Both continuations reach floor 100 (`4106–4145`). The floor-100 win triggers the ending (`4078–4079`). The adapter preserves those transitions and records actual use of the special shop separately from the chosen branch.

Training SQLite stores each eligible transition with its current and next heads, FP32 features, packed masks, action, reward, terminal flag, Combat/Reward version, source and cycle. The final transition becomes available immediately when the milestone or defeat resolves. Fixed evaluation generates telemetry only. The first random segment records every actual visit. Its replay importer rejects backward stage movement, reopened stat menus, excessive modifications, and any shadow verification mismatch. For compatible traces it maps only observed edits, maps procedural head finishes without resource changes, and compares the shop-exit player state, gold, reroll cost and reroll RNG with the real trace. Outcome comes exclusively from the real run. Earlier-run data is outside this replay buffer.

The default replay uses the most recent 32,768 transitions and at most two prior cycles. It draws 75% of its sample attempts from the newest quarter of the eligible window. It prunes expired records at checkpoint opportunities, so records since the last save can temporarily extend the stored table. Save watermarks align replay and telemetry with the model. `latest.pt` resumes the full state. Archived boundary checkpoints preserve policy snapshots and may refer to replay rows that have aged out.

| Setting | Initial value and rationale |
|---|---|
| Shared widths and optimizer | 128/128 tanh and Adam at 0.001. A compact shared representation keeps per-head scoring inexpensive. |
| Loss, discount and target refresh | Huber, discount 1, hard target copy every 250 updates. Huber limits the impact of early large errors without changing terminal labels. |
| Batch and update cadence | 128 rows per batch and roughly one update per 128 new replay entries during Shop training, capped at eight updates per collection rollout. Up to 16 bootstrap updates use eligible random-segment data. |
| Exploration | Linear decay from 1.0 to 0.10 over 500,000 DQN training decisions. Evaluation and frozen-Shop training use greedy choice. Exploration samples uniformly from the legal actions. |
| Runtime | Twelve live game slots over six grouped Windows workers by default. Batched CPU inference groups ready states by head, and CUDA applies PPO and DQN updates. |

These are starting values. Rare high-floor transitions and the mixture of current and earlier Combat/Reward versions can affect calibration. The action-space ordering restricts revisiting a previous stage by design. A greedy policy can still repeat edits within a stage until the budget is exhausted.

## Minimum validation and local benchmark

The isolated CUDA run used eight episodes, two four-episode segments and the actual pretrained PPO. It processed 1,146 primitive actions and evaluated two games at each of steps 0, 4 and 8. The first segment reused three verified uniform transitions from one visit and rejected three incompatible visits. The Shop segment made 175 DQN decisions and performed 12 updates. The boundary checkpoints show that only the active model's weights moved. Existing run artifacts retained their original hashes.

On one recorded floor-10 state, the old 16-candidate implementation took a median 2.76 ms to generate proposals and 0.10 ms to score them. The new DQN encoded the first decision in 0.06 ms and scored its entire legal head in 0.09 ms. A complete untrained DQN visit at epsilon 0.15 took 43.86 ms and a median 161.5 policy decisions. The old complete plan-and-apply loop took 4.36 ms in a separate measurement of the same starting shop. A DQN decision is cheaper than generation of all old candidates, while this untrained policy used more decisions per visit. The microbenchmark does not establish learned-policy speed or sustained training throughput. Reproduce the revised measurement with `python -m alternating_rl.benchmark`. The old implementation is available at commit `74ad0ab`.

## Run on Windows

Use the existing project `.venv` and the trusted local Phase 3B-2 checkpoint. Double-click `run_alternating_rl.bat` for a fresh run in `data/alternating_ppo_dqn`. The previous candidate-based folder remains separate. Automatic resume from a candidate checkpoint is unsupported because observations, actions, learner and replay state differ.

```bat
validate_alternating_rl.bat
run_alternating_rl.bat
resume_alternating_rl.bat
report_alternating_rl.bat
```

The launchers accept `--config` and `--output`. For initialization only, `--initial` can point to the trusted 1M PPO checkpoint. Resume restores the two online models, DQN target, both optimizers, live games, replay and metrics watermarks, and RNGs. The English report records action distributions, head budgets, repeat and inverse edits, economy and loadout outcomes, reroll changes, calibration, exploration, event branches, losses and computation metrics. Raw SQLite data and checkpoints stay local. The full million-episode training was left for the local run.
