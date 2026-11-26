# Phase 3C · Local reinforcement learning implementation

The Windows launchers run independent Python processes using the project `.venv`. The simulator supplies the rules and transitions. The RL adapter supplies observations, actions, rewards and episode limits. The first learner is masked PPO with one shared encoder, Combat, Reward, Shop and Event policy heads, and one shared critic. [Environment design](rl_environment_design.md) · [Simulator guide](simulator.md)

## Windows setup and commands

From the repository root, use the existing project virtual environment with the CUDA-enabled PyTorch installation described in the local benchmark. If setting up a fresh machine, create it with `py -3.12 -m venv .venv`, install a compatible official CUDA PyTorch wheel into it, and install the RL dependencies. Select `cpu` in the configuration to use a CPU PyTorch installation. [Hardware benchmark and PyTorch setup](hardware_benchmark.md)

```bat
.venv\Scripts\python.exe -m pip install -r rl\requirements.txt
run_rl_validation.bat
```

| Launcher | Behavior |
|---|---|
| `run_rl_validation.bat` | Runs the small RL tests, including real updates on CPU and on CUDA when available. It writes integration fixtures to a temporary directory. |
| `run_training.bat` | Starts a new strategic-shop run using `rl/shop_training.json` and `data/rl_shop_strategic`. A fresh experiment requires a fresh output directory. |
| `resume_training.bat` | Restores the same run's `latest.pt` and database. Legacy runs use `rl/training.json` by default. Strategic runs use `--config rl\shop_training.json`. Raise the cumulative budget before resuming a run that used it. |
| `run_evaluation.bat` | Evaluates the saved model and Uniform Random Policy on the same independent test seeds and with the same action limit. Use `--split validation` for exploratory evaluation. Reserve the test split for a selected checkpoint. |
| `generate_training_report.bat` | Regenerates English Markdown and Matplotlib charts from available local training, validation and test databases. |

Double-click the launchers for the defaults, or use CMD to supply a different JSON file, run directory or curriculum stage. Example commands for a separate run are below. Copy and edit `rl/shop_training.json` for a strategic run. [Strategic shop policy and evaluation](shop_policy_redesign.md)

```bat
run_training.bat --config my_training.json --output data\trial_01
resume_training.bat --config my_training.json --output data\trial_01 --stage 3
run_evaluation.bat --config my_training.json --output data\trial_01 --split test
generate_training_report.bat --config my_training.json --output data\trial_01
```

Checkpoints and databases are local. Load only checkpoints from your own trusted run directory. The save format contains Python game objects as well as tensor weights. Keep the checkout and runtime versions used to create them. The output directory is locked during training and checkpoint evaluation. Inspect `errors.log` after an unexpected exit. Interrupting the process or losing power can discard progress since the last completed checkpoint. The matching SQLite watermarks roll back any training records beyond that checkpoint on resume. Standalone evaluations have separate databases.

## Environment and leakage control

`rl.environment.DungeonEnv` implements Gymnasium `reset` and `step` with a fixed FP32 feature vector, a discrete phase and an action mask. The legacy profile has 1,088 features and 667 action IDs. The strategic profile has 2,234 features and 416 action IDs. The environment encodes visible actors, stats, HP and substitute bars, status icon presence, skill inventory, ordered slots, current uses, rewards, shop controls and tracked history. Hidden body HP under a substitute uses a last-seen value and age. Learned skill costs and purchase flags represent trackable inventory changes. Enemy internal actions, future encounters, reward pools, exact status counters, raw IDs and RNG state stay outside the model. The adapter projects observations before crossing the worker boundary. It uses public availability rules for masking and validates the available mappings against the simulator in tests. [Observation inventory](observation_space.md) · [Strategic shop interface](shop_policy_redesign.md)

One step is one accepted game decision. Termination means clear or death. Truncation means the action cap or the separately configurable confirmed Mimic–Counter lock. The detector gates on verified conditions including the single forced Counter action, priority order, Mimic's available move, positive HP, and absence of substitutes and active statuses. Privileged fields used to certify the lock stay outside policy features and masks. Independent evaluation retains the 10,000-action limit, logs the confirmed lock alongside a faithful timeout, and counts both as non-clears. The learner stops GAE continuation at episode boundaries, bootstraps ordinary time limits from the last observation, and uses zero continuation for death, clear and a proven absorbing lock. [Mimic timeout analysis](mimic_timeout_analysis.md)

## PPO, curriculum and defaults

The network uses a shared tanh encoder and critic. The active phase's head scores its corresponding catalog IDs. Masked categorical sampling, entropy and PPO likelihood ratios use only legal choices. A bounded CPU rollout buffer stores projected batches. The central learner collects from 12 Windows `spawn` workers by default and uses one CUDA device. `ppo.rollout_size` counts decision rows per worker. The default 128 rows with 12 workers yield up to 1,536 transitions per update. Reports, checkpoints and model inference run in the user's local process and workers.

| Setting | Initial value and rationale |
|---|---|
| `hidden_dims` | `[128, 128]`. The list controls both the number and widths of layers. This is a modest initial footprint for the shared encoder. |
| `learning_rate`, `clip_range`, `update_epochs` | `0.0003`, `0.2`, and 4. Conservative initial PPO update controls for the first experiment. |
| `gamma`, `gae_lambda` | `0.999` and `0.95`. A long reward horizon helps retain the delayed value of Gold, Exchange and shop investments. GAE controls the variance of the resulting estimates. |
| `entropy_coef` | `0.01`. Stochastic masked training and the entropy term encourage exploration, including early reward choices that pay off later. |
| `batch_size`, `rollout_size` | 256 transitions per minibatch and up to 128 decisions per worker per rollout. Start with bounded memory and measure actual throughput before increasing either. |
| `budget` | 1,000,000 accepted decisions. Supply `steps`, `episodes`, or `seconds` and use `null` to disable a limit. The run stops at the first applicable limit, checked between worker batches. A batch may cross the decision or episode limit, and an update can finish after a time limit. Evaluation and checkpoint time are separate from the training-time budget. |

These values are starting choices, with throughput, reward weights and advancement thresholds to be assessed from independent evaluation. The schedule below uses one-time target bonuses and a small bonus for each newly reached floor. Death defaults to zero reward and clear pays `+1` in every stage. Stage 9 uses clear-only reward. Each target remains achieved for the remainder of its episode, and play continues through the normal outcome or cap.

| Stage | Target | Validation rate to advance or flag |
|---:|---|---:|
| 1 | Reach the next floor | 85% |
| 2 | Defeat the floor 5 boss | 65% |
| 3 | Defeat the floor 10 boss | 55% |
| 4 | Defeat the floor 20 boss | 45% |
| 5 | Reach floor 40 | 40% |
| 6 | Reach floor 60 | 35% |
| 7 | Reach floor 80 | 30% |
| 8 | Clear the game | 20% |
| 9 | Optimize clear rate | 90% milestone |

Combat and reward naturally account for most early decisions. All reachable heads learn throughout the curriculum, and longer runs supply more shop and event decisions. Periodic validation can advance the stage automatically. `--stage N` allows a manual choice. Stage changes preserve network and optimizer state, write checkpoints on both sides, start fresh worker episodes and log the old unfinished games outside completed-episode counts. Ordinary resume at the same stage restores active games and RNG state. It requires matching workers, master seed, simulator settings, curriculum and network dimensions. Start a new experiment directory for a different curriculum or architecture. Progress and rewards never use the random baseline's skill correlations.

## Storage, evaluation and reports

The default run directory contains `training.sqlite3`, `latest.pt`, stage checkpoints, and a `report` subdirectory. `run_evaluation.bat` produces a separate evaluation database and report directory. The SQLite writer commits bounded batches and stores episode rows, update summaries, evaluation summaries and events. Full per-step states are limited to live games and the fixed-size rollout, with active state saved for resume. Checkpoints save model, optimizer, active games, cumulative counters and Python, NumPy, Torch and available CUDA RNG states. Logged seeds are derived from distinct run, validation, test and random-policy domains. The test panel remains separate from stage promotion.

Training output reports cumulative steps, episodes, training time and throughput. The databases and generated English reports include clear and target success rates, mean, median and maximum floor, floor-wise survival, episode reward and actions, policy and value loss, phase counts and entropy, timeout and early-deadlock rates, elapsed training time and hardware telemetry. Charts use environment steps, completed episodes and elapsed time. NVIDIA metrics come from `nvidia-smi` when available. Clear rate is reported with a Wilson interval. Paired evaluation uses the same game seeds, legal decision semantics and cap for PPO and uniform random. The historical million-game run is context because it used a different seed panel. [Million-game baseline](baseline_random_1m_results.md)

Evaluation runs a separate CPU worker pool, batches masked greedy inference, and displays completed episodes, actions, the longest active episode, elapsed time and ETA. Training prints its own elapsed time and cumulative validation time separately. The learner saves `latest.pt` immediately before scheduled validation. If interrupted during validation, resume reruns the pending panel from that checkpoint. `run_evaluation.bat --split validation --policy ppo` measures the saved policy alone. The default standalone comparison still runs PPO and uniform random on paired seeds. [Evaluation investigation and benchmark](rl_evaluation_performance.md)

## Phase 3C validation scope

Validation uses seeded engine comparisons, hidden-field perturbations, fixed reward and deadlock fixtures, tiny PPO updates on CPU and CUDA, and an eight-decision CPU integration run with two spawned workers. The integration run exercises stage promotion, checkpoint resume, an independent matched evaluation, SQLite writes and report generation. This stage establishes functionality. Full-scale throughput, learned clear rate, long-run reliability and Steam parity require later evidence.
