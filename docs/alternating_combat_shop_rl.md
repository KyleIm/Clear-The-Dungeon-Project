# Phase 3C · Alternating PPO and Shop Value Optimization

This experiment starts from the final Phase 3B-2 Combat and Reward weights and Adam state. It runs in a separate directory and uses newly generated data. The existing 661,093 shop exits are outside its replay dataset. The Python simulator supplies the original legal actions and transition rules. [Phase 3B-2 results](combat_rl_1m/training_report.md)

## Schedule and fixed evaluation

The default adds 1,000,000 games in 40 segments of exactly 25,000 completed episodes. Each pair of segments is one cycle. Odd segments update Combat and Reward PPO, and even segments update only the Shop Value Network. No game remains active across a segment boundary. Each model and its optimizer carry forward. The schedule allocates 500,000 episode opportunities to each learner, including games that never reach a shop.

The initial 25,000 games keep the original Uniform Random Shop Policy, including incomplete allocations and random spending. Those exits and their outcomes seed the first Shop update. At the start of the second segment, the value learner performs up to 16 initial minibatch updates from the newly recorded data. During later Combat and Reward segments it is frozen and chooses the highest scoring candidate. During Shop segments it chooses a uniformly sampled candidate with probability 0.15 and the highest scoring candidate otherwise. If there are still no resolved labels after the initial segment, candidate choices remain uniform until a value update is possible. All training segments continue collecting shop samples. No externally prescribed stat or skill targets are provided.

At step zero the pretrained PPO policy is evaluated with uniform random shops and events. After each segment the current models are evaluated with masked argmax Combat and Reward decisions, greedy candidate selection once a Shop model exists, and uniform events. The 128 seed pairs reproduce the Phase 3B-2 evaluation manifest by default. The training RNG domains are separate. Evaluation rows never enter the Shop replay pool. These seed pairs enable checkpoint comparisons. Repeated panels remain one fixed set of scenarios. The reports also include the historical 1M random, PPO 100K and PPO 1M results. [100K results](combat_rl_100k/training_report.md) · [Historical uniform baseline](baseline_random_1m_results.md)

## Combat and Reward learner

The model retains 583 inputs, shared tanh layers of 128 and 128 units, the seven-output Combat head, five-output Reward head and shared critic. It uses the same Masked PPO update and hyperparameters from the verified 1M checkpoint. Each new floor adds one reward. An automatic shop or event transition contributes its progress to the preceding controlled decision. Shop spending and skill selection add no separate reward or penalty. The default keeps 12 game slots on six grouped `spawn` workers, batched CPU inference and CUDA PPO updates. The next value is reused at the next actor call under the same weights, with separate bootstrap at the rollout tail. [Combat and Reward spec](combat_reward_rl.md) · [Optimized runtime](combat_reward_performance.md)

## Shop input specification

The model receives exactly **671 FP32 inputs** from the state immediately before `leave_shop`. Entry state constrains the candidate planner and is recorded separately.

| Inputs | Nodes | Encoding |
|---|---:|---|
| Player | 26 | Current and maximum HP, current and base ATK/DEF/SPD, ACC/EVA/DR, then four type bits and 11 status-presence bits. The ordering is inherited from the Combat encoder. |
| Six equipped slots | 450 | Each slot uses the 74 existing descriptor and use fields, then current mana cost. Locked slots are all zero. An equipped skill remains available as a descriptor with allocation zero when it has no uses. |
| Acquired skill flags | 88 | Multi-hot in ascending static skill ID order. |
| Skill mana costs | 88 | In the same ID order, with zero for unlearned skills. |
| Maximum, used and remaining mana | 3 | Used mana is the sum of current allotment × current cost. |
| Unused stat points and gold | 2 | Current unspent values. |
| Floor and region | 12 | Floor divided by 100 and the registry-ordered 11-region one-hot. Normal shops explicitly record the preceding region because the engine changes the current area to `Shop`. The floor-99 shop keeps its region. No future region draw enters the encoding. |
| Event choice | 2 | Selected route A `[1,0]`, selected B `[0,1]`, or `[0,0]` before the choice. This records the player's choice separately from the engine's effective path. |
| **Total** | **671** | |

Numeric player features, unused points and mana use signed `log1p` with unit 100. Skill uses and current mana costs use unit 10, gold uses unit 1,000, and power and accuracy use division by 100. Priority is the original priority mapping, and the 65 skill category bits use ascending registered category IDs. The 88 registry entries include `strike`, marked dummy in the source, and Sacrifice. Neither appears in the static learnable union of initial pools, monster decks and upgrade targets, which contains 86 entries. Sacrifice is the combat fallback and has a zero acquired flag and cost feature in ordinary play. Keeping both registry positions preserves the requested 671 inputs.

The type bit order is Normal, Beast, Elemental, Undead. The status bit order is stun, poison, sprout, regeneration, target, sleep, reverse, aurora, immunity, imitation and helmet. An occupied slot is encoded in this order: occupancy, allotted uses, type bits, power, accuracy, priority, category bits, current mana cost.

### Ren’Py checks

The comparison used the original source snapshot documented in the simulator fidelity audit. `script.rpy:115,202,651` identifies the dummy and Sacrifice entries. `319,1517–1518` gives maximum HP as twice base HP. `334,357–364,1546–1578` covers mana, allocation and one-time cost reduction. `1582` onward covers point redistribution and stat minimums. `1690–1740` covers key unlocks and mana potions. `1753–1785,4344` covers skill replacement and saved allocations. `4014,4067` covers learning the offered enemy skill. `4297–4300` distinguishes normal shop region handling from the floor-99 exception. [Source fidelity audit](simulator_fidelity.md)

The source choice at floor 35 (`3823–3870`) assigns A to Pickpocket and B to Help. The experiment samples the two legal event choices with equal probability. It records chosen route and decision-time version, effective path, actual floor-99 shop entry and exit, and final outcomes. At floor 99 (`4106–4145`) the initial Gate victory proceeds to a second Ella encounter on the effective A path, or opens the special shop on the effective B path. Both continuations lead to floor 100. The source ending branch fires on the final victory (`4078–4079`). The adapter tests both simulator paths with controlled KOs. Installed Steam runtime parity remains unverified.

## Candidate planning and value targets

The planner always offers immediate exit. Other candidates use finite random sequences of legal shop edits on a copy of player and shop state. It uses original menu availability and shop actions for stat changes, equipment, cost reduction, allocations, merchandise and keys. It can leave points, mana or gold unused. It excludes stochastic rerolls from scored plans to avoid evaluating unseen reroll outcomes. Random shops in the initial segment retain the original reroll action. Identical encoded candidates are deduplicated. Neither battle simulation nor the live game RNG is used for candidate planning. Only the selected legal action sequence is applied to the real simulator. The engine action cap counts those primitive actions. The default `uniform_menu` gives viable menus equal proposal probability, independent of their numbers of buttons. The alternative `uniform_action` weights each menu by its count of legal edits. Both draw edits uniformly inside the chosen menu.

The CPU value network has hidden widths 128 and 64 with tanh, followed by one logit and a sigmoid probability. Adam uses learning rate 0.001 and minibatches up to 128. Training uses numerically stable binary cross-entropy with logits. The model shares information across shops but receives floor and build context. It uses no success weighting so the target retains an observed probability interpretation. Candidates from all ready games in a worker round are combined into one CPU scoring call. Defaults are 16 proposed candidates and up to 24 edits per plan. More candidates increase coverage at a measurable planning cost.

| Shop | First state that confirms success |
|---|---|
| Floors 10–80 | Entry into the next regular shop ten floors later, after defeating that area's boss. |
| Floor 90 | Actual entry to floor 100 after all required floor-99 encounters and any special shop. |
| Floor 99 | The game reaches its cleared state after the final boss. |

Each exit's pending features remain attached to the active episode until the outcome resolves. The adapter emits a label immediately at the milestone or at defeat. Success is 1 and defeat before success is 0. An unresolved action-limit timeout is recorded with a null target and excluded from BCE and calibration. If the last allowed action reaches the milestone, that successful label is retained. Samples include the exit and resolution Combat/Reward versions and cycle. Training samples generated during PPO updates can span several versions, while Shop segments hold Combat/Reward fixed throughout.

SQLite keeps the samples, action records and report data under one checkpoint watermark. Each Shop segment loads the latest 4,096 eligible rows ordered by exit policy version and insertion order, then adds newly resolved rows to the bounded in-memory window as they arrive. Sampling within it gives the newest row up to 16 times the weight of the oldest. A Shop update runs after each collection rollout when data exists. Per-shop and version summaries expose rare success classes. Calibration compares predicted probabilities with resolved outcomes, both within prediction bins and by floor. Greedy selection and action-limit censoring can affect the observed calibration sample. In particular, a learned value trained under sampled Combat/Reward actions is evaluated alongside greedy Combat/Reward actions. Sparse high floors and unusual candidate combinations can have unreliable scores until explored.

## Local measurements and validation

A 30-repeat microbenchmark used one recorded floor-10 shop with fixed starting state, 24 allowed edits, and CPU scoring. Times are medians per batch. To repeat it, run `python -m alternating_rl.benchmark`.

| Candidate budget | Mean unique candidates | Planning | Batched scoring |
|---:|---:|---:|---:|
| 8 | 8.00 | 1.34 ms | 0.07 ms |
| 16 | 15.97 | 3.00 ms | 0.15 ms |
| 24 | 23.93 | 4.27 ms | 0.12 ms |

The benchmark measured planning and inference without running candidate fights. Candidate planning dominated scoring in this fixture. Its cost can vary with the shop inventory. A Windows CUDA smoke run used the existing 1M model, two four-episode segments and a 512-action cap. It completed 998 training actions, 6 evaluation games across step zero and both boundaries, and seven Shop updates including two warm-up updates. It recorded seven training shop labels, all negative in this small fixture. Separate fixtures covered a positive milestone, route-specific floor-99 play, legal candidate application, model updates, frozen weights, active-game restoration and checkpoint-aligned SQLite rollback. Checkpoint hashes for the earlier 100K and 1M runs were unchanged. Sustained throughput and clearance improvement remain unmeasured.

The step-zero CUDA checkpoint matched the source Combat/Reward weights exactly. At the first boundary only Combat/Reward weights had changed. At the second boundary the Combat/Reward weights still matched the first boundary and Shop weights had changed. The full simulator and learner suite passed 102 tests.

## Windows launchers

Use the project `.venv` and existing Combat RL dependencies. A new run expects the trusted local Phase 3B-2 `latest.pt`. Its default output is `data/alternating_combat_shop`. The launcher refuses to start a fresh run in an occupied experiment directory. `--config`, `--output` and `--initial` can select alternatives. Resume restores both optimizers, Torch CPU/CUDA RNG, exploration and sample RNG, live games and their RNGs, and database watermarks. Boundary checkpoints are archived separately. The source checkpoint is read-only and is required only for initialization.

```bat
validate_alternating_rl.bat
run_alternating_rl.bat
resume_alternating_rl.bat
report_alternating_rl.bat
```

The report launcher exports floor and outcome curves, reward selection trends, shop allocations and spending, calibration, event route cohorts, resource samples, and performance timers. Databases and checkpoints remain local. This work did not launch the million-episode training run.
