# Phase 3D · Shop policy redesign

The greedy Phase 3C policy could spend the full action budget repeatedly swapping skills, changing stat allocations, or opening and closing a shop menu. All 87 action-limit timeouts in the million-game uniform random baseline occurred in Mimic combat. Those results describe different policies and show why shop loops need separate metrics. [Evaluation investigation](rl_evaluation_performance.md) · [Mimic timeout analysis](mimic_timeout_analysis.md) · [RL environment design](rl_environment_design.md)

## Strategic shop interface

The strategic profile uses five learned decisions at the top of the shop. Skill Configuration edits a full loadout and allocates its uses within the existing mana and slot limits. Skill Cost Reduction selects an eligible learned skill. Stat Redistribution edits a four-stat allocation while conserving the original point budget and minimum. Merchandise Purchase selects an eligible transaction at its existing price and prerequisites. Leave Shop saves the configuration and advances using the existing game rule.

The adapter carries out original menu transitions and configuration operations. Configuration drafts expose the starting state and the prospective state to the policy, including equipped skills, locked or unlocked slots, uses, costs, spent and available mana, main stats and available points. A sequential draft avoids enumerating all complete loadouts or point distributions. Gold prices, affordability and costs remain game data. The model can learn spending trade-offs from outcomes without fixed item or skill bonuses. Shop visits retain an explicit Leave decision. The simulator's source files and transition rules are unchanged.

Skill drafts pick each unlocked slot in order, then choose its use count from the remaining mana. If a swap requires a temporary unequipped skill, the adapter uses an eligible spare to execute the original replacement commands. With no spare, the original replacement rules preserve existing slot order. Stat drafts assign each of ATK, DEF, HP and SPD in sequence, with decimal choices for any allowed allocation and shortcuts for retaining a value or spending the remaining points. The adapter releases points before allocating them through the original commands. The policy can keep the starting configuration. After any committed task it can choose its next strategic action, including another purchase or another configuration.

The shared encoder, Combat, Reward and Event heads, and shared critic retain their architecture. The strategic profile has 2,234 features and 416 action IDs. The legacy profile keeps its 1,088 features and 667 IDs. Changes to the encoder input and Shop Head output shapes make the checkpoints incompatible across profiles. The runner rejects a cross-profile resume before restoring the run database.

## Reward and evaluation

For each visit the adapter starts four independent function counts at zero. The first selection of each function is free. Each later selection of that same function adds `-epsilon` to the reward returned to PPO. The total cost is `-epsilon × sum(max(0, count - 1))`. Draft editing within a selected function and Leave carry zero repeat cost. The counts are policy observations. The default `shop_repeat_epsilon` is `0.0001` in each curriculum stage and can be set to `0` in a separate control run with the same strategic interface. The cost does not alter gold or other game resources.

The new profile defaults to seeded stochastic intermediate validation. An external setting selects deterministic validation. Both evaluation modes use the same checkpoint, game seed panel, action cap and observation schema. Argmax selects the highest legal probability. Stochastic sampling draws from the same masked distribution with an independent per-episode policy seed, so worker completion order does not consume another episode's random draws. The evaluator retains parallel CPU environments, CUDA batching, progress and ETA, separate timing, and the pre-validation checkpoint.

An environment step is a policy decision, including draft choices. A committed draft may execute several `Game.step()` calls in one environment step. The action cap counts policy decisions in both evaluation modes. Shop telemetry also counts actual `Game.step()` calls, including the original navigation handled by the adapter. Averages for strategic shop actions count the five top-level functions and exclude draft choices. Shop timeout and exit rates use recorded shop visits as their denominator.

Shop repeat rates, per-visit spending, purchase and configuration outcomes, and shop reach and exit rates are needed to distinguish reduced waste from avoiding useful investments. A comparison of argmax and sampling from one checkpoint measures the evaluation modes. To isolate learning effects of the repeat cost, train paired runs of the same strategic profile with identical seed manifests and budgets and compare `epsilon=0.0001` against `epsilon=0`. The small functional validation supplies no long-run estimate of that effect.

## Decision statistics

SQLite stores one aggregate `decision_stats` row per evaluation, linked to the evaluation ID. The generated `decision_statistics.md` shows availability and actual selections separately. Combat tables use per-skill availability denominators and average the masked probabilities over eligible decisions. Reward breakdowns include floor and checkpoint trends. Shop tables report available and selected functions, first and repeated choices, cumulative penalties, configuration and stat changes, purchases and net gold spending. Visit summaries include Leave probability, strategic and actual engine actions, exits and timeouts. Event tables link the choice to later floor and game outcomes. Configuration examples are bounded alongside streaming totals. Detailed trajectories stay out of the database.

## Local launchers

- `run_training.bat` starts the new strategic profile from `rl/shop_training.json` and writes to `data/rl_shop_strategic` by default.
- `resume_training.bat --config rl\shop_training.json` resumes the strategic profile. `resume_training.bat` without arguments retains the existing legacy configuration in `rl/training.json`.
- `run_evaluation.bat --config rl\shop_training.json --split validation --policy compare` evaluates deterministic and stochastic PPO on the same game seed panel. Set `evaluation.validation_episodes` in a copy of the config to select the panel size.
- `generate_training_report.bat --config rl\shop_training.json` regenerates reports. `run_rl_validation.bat` runs the compact automated tests.

Use `--config` and `--output` for independent experiments. The strategic and legacy run directories remain separate. [Local setup and launchers](rl_implementation.md)

## Local CUDA validation

The isolated run used a 10,000-step budget, validation every 2,000 steps, 8 episodes per validation, 12 CPU workers, CUDA and auto-advance disabled. Stage 1 used `shop_repeat_epsilon=0.0001`. The validation copy used 42 rollout decisions per worker to check close to each scheduled threshold. Training finished at **10,008** policy steps after the last 12-worker batch, with **458 completed training episodes**. Training took **14.3 seconds** and the four separate validation panels took **3.1 seconds** in total. The panels at 2,016, 4,032, 6,048 and 8,064 steps all finished with zero timeouts. At 8,064 steps one run entered the shop, repeated redistribution once, recorded `-0.0001`, and exited. The earlier checkpoints and databases were preserved.

The CUDA evaluator then ran each mode once on the same 128 game seeds with the saved 10,008-step checkpoint and 10,000-decision cap. Neither mode recorded a clear. The results characterize a short Stage 1 run.

| Metric | Deterministic argmax | Seeded stochastic |
|---|---:|---:|
| Mean furthest floor | 8.70 | 6.02 |
| Furthest floor | 10 | 13 |
| Episodes reaching a shop | 54/128 (42.19%) | 7/128 (5.47%) |
| Shop exits per visit | 0/54 (0%) | 7/7 (100%) |
| Shop timeouts per visit | 54/54 (100%) | 0/7 (0%) |
| Episodes ending in a shop timeout | 54/128 (42.19%) | 0/128 (0%) |
| Mean strategic actions per shop visit | 766.72 | 6.43 |
| Mean primitive engine actions per shop visit | 74.00 | 212.00 |
| Shop visits with a repeat | 54/54 (100%) | 5/7 (71.43%) |
| Repeat selections | 41,349 | 24 |
| Cumulative shop penalty | −4.1349 | −0.0024 |
| Mean Leave probability when available | 18.06% | 18.74% |
| Total policy decisions | 542,503 | 3,358 |
| Evaluation wall time | 101.83s | 1.91s |

Argmax repeatedly opened Stat Redistribution. Only 54 of its 41,367 completed redistribution drafts changed the recorded build. It never selected Leave, despite the nonzero Leave probability. The sequential draft removed back-and-forth navigation within a task, while the model could still repeatedly choose the same strategic task. In the stochastic panel, six of seven repeated skill configurations changed the recorded build and one repeated merchandise purchase succeeded. It spent 3,100 gold on merchandise and 1,000 on a skill cost reduction. The higher primitive action count per stochastic visit reflects actual edits being applied. These observations show that the interface still permits repeated investments with game effects. A matched zero-penalty training control was configured as an option and was outside this validation run. The different shop reach rates and small number of stochastic shop visits leave shop avoidance and the penalty's causal effect unverified. Neither panel reached the floor-35 event, so event statistics were verified with automated fixtures.

The existing Phase 3C checkpoint at 145,920 steps loaded with the legacy network shape and advanced one saved environment in memory. The original run artifacts remained in place.

To reproduce the functional run, copy `rl/shop_training.json`, set the step budget to 10,000, `ppo.rollout_size` to 42, `evaluation.interval_steps` to 2,000, `evaluation.validation_episodes` to 8 and `evaluation.auto_advance` to `false`. Use a new output directory. After training, set the copied validation count to 128 and run the paired comparison on that directory. The training and evaluation SQLite files, checkpoints and generated charts remain local.
