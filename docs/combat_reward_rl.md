# Phase 3B · Combat and Reward reinforcement learning

This experiment optimizes subsequent floor progress in the Python simulator. The default local run completes **1,000,000 episodes**. Every episode starts from the original game initialization. The same reward function applies throughout the run. The engine supplies legal actions and transitions for all phases.

## Input and actions

| Feature group | Nodes | Encoding |
|---|---:|---|
| Player and monster | 52 | Each actor has 11 numeric values for current and maximum HP, current ATK/DEF/SPD, base ATK/DEF/SPD, ACC/EVA/DR, then four type bits and 11 active-condition bits. |
| Six equipped slots | 444 | Each slot has availability, remaining uses, four type bits, power, accuracy, priority and 65 category bits. A locked slot is zeroed. An unlocked but exhausted slot keeps its descriptor and is masked out of combat. |
| Floor, region, and gold | 13 | Floor divided by 100, an 11-region one-hot vector in registry order, and scaled gold. |
| Reward skill | 73 | A single skill descriptor in slot format without uses. Missing offers and combat observations zero this block. |
| Phase | 1 | Combat is 0 and Reward is 1. Reward observations zero the monster block. |
| **Total** | **583** | FP32 features. The phase and legality mask also route the output independently. |

Actor numerics use signed `log1p` scaling with unit 100, allotted uses use unit 10, gold uses unit 1,000, and power and accuracy are divided by 100. Priority comes from the simulator's current skill category rules. Category one-hot positions are the 65 distinct numeric IDs in ascending order, including gaps in the IDs. The input includes simulator base stats as specified. Steam observation equivalence has not been established.

Combat has seven outputs for slots 1–6 and Sacrifice. Reward has five outputs for Gold, Potion, Initialize, Stat and Exchange. The active head scores only the phase's legal actions. Shop and event moves use an independent per-episode Python `Random` stream and choose directly from `Game.get_legal_actions()` in engine order. When the entire game is controlled by uniform random, the same seed stream samples across all phases, which reproduces the baseline policy's draw order. Early shop exits, incomplete allocations and reversible shop choices remain legal. The existing engine-action cap bounds the complete episode, including automatically sampled moves.

## Optimizer and fixed reward

Masked PPO uses a shared `583 → 128 → 128` tanh encoder, two phase-specific linear policy heads and a shared scalar critic. The objective uses the clipped probability ratio, one entropy bonus and a value regression term. Generalized advantage estimation follows each worker's episodes and stops at all episode boundaries, including capped episodes. The critic bootstraps ongoing trajectories at rollout boundaries. The supervised value coefficient and maximum gradient norm are both 0.5.

Each new floor adds **+1** reward. A controlled action includes any subsequent uniformly random shop or event actions before the next Combat or Reward decision. All floor progress during that interval is credited to that transition. Death, timeout, clearing, spending gold and using any particular skill have no separate bonuses or penalties. At the episode level, undiscounted reward equals furthest floor minus the starting floor. A clear and a defeat on the same final floor can therefore earn the same reward. Clear rate is tracked separately. Discounting is per controlled transition, so the learning objective mildly favors earlier progression.

| Setting | Initial choice and rationale |
|---|---|
| Hidden widths, Adam learning rate | `[128, 128]` and `0.0003`. These give a modest common encoder and a conservative default optimizer step. |
| Discount and GAE lambda | `0.999` and `0.95`. Long-horizon progress matters, and GAE reduces the variance of return estimates. |
| Entropy coefficient and PPO clip | `0.01` and `0.2`. They encourage exploration and bound large policy-ratio changes. |
| Rollout and update | 128 decision rounds across up to 12 CPU workers, 256 transitions per minibatch, and four update epochs. Only one rollout is kept in training memory. |
| Termination and evaluation | 10,000 engine actions per game. Evaluation starts every 25,000 completed training episodes at a worker-round boundary and uses 128 fixed seeds per policy. A final panel runs at the budget boundary. |

These are initial experiment settings. [Local hardware benchmark](hardware_benchmark.md)

## Run on Windows

Use the project `.venv` with a CUDA-enabled PyTorch wheel and install the optional analysis dependencies. Set `device` to `cpu` in a copied settings file for a CPU-only machine.

```bat
.venv\Scripts\python.exe -m pip install -r combat_rl\requirements.txt
validate_combat_rl.bat
run_combat_rl.bat
resume_combat_rl.bat
report_combat_rl.bat
```

The training, resume and report launchers accept `--config` and `--output`. The defaults are `combat_rl/training.json` and `data/combat_reward_rl`. A fresh run requires a new directory. Resume uses the same directory and settings. Increase the cumulative episode budget in the config if the existing run already reached its budget. The model, optimizer, RNG states and live simulator episodes are stored in `latest.pt`. Only open checkpoints from trusted local runs. On interruption, resume rolls SQLite back to the last saved checkpoint and may repeat a pending evaluation. Raw databases, checkpoints and generated run artifacts stay local.

## Evaluation and reporting

Evaluation uses separate fixed seeds from training. The saved model uses masked argmax, and the fully uniform policy is evaluated on the same game seeds and action cap. The historical million-game distribution provides an additional reference. That historical panel has different seeds and is reported separately. Checkpoint evaluation, raw engine action count, completed games, rollout losses and throughput are logged. SQLite stores per-episode floor, outcome, defeat opponent or timeout phase, Combat and Reward selections, and each random shop and event action. It also records the ordered shop-exit allocation, including unused stat points and mana. The report aggregates floor reach and distribution, mean, median and maximum floor, clear rate, action distributions, timeout and defeat causes and learning curves.

Random shop choices can leave skill charges at zero or redistribute stats without spending the available points. These runs may remain limited by shop configuration as the learned Combat and Reward policies improve. Historical associations from the random experiment are available for interpretation. [Shop exit analysis](shop_exit_replay/report.md) · [Million-game random results](baseline_random_1m_results.md)

## Functional validation

The Windows CUDA launcher completed four capped training episodes and two fixed-seed episodes per evaluation policy. The fixture used two workers and a four-action cap. The separate CPU integration test covered checkpoint resume and SQLite counts. Unit tests checked the encoder and masks, CPU and CUDA parameter changes, and exact uniform-policy replay of a previously recorded shop-reaching episode, including its shop choices and exit allocation. These checks establish functionality. The million-episode run and learned policy performance remain for local execution.
