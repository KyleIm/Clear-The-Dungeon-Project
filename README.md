# Clear The Dungeon Project

A local research project for Dungeon & Rogue, a roguelike on Steam. The goal is to achieve at least a **90% clear rate in the Steam game**. The latest simulator baseline recorded **0 clears in 1,000,000 uniform-random runs** and reached floor 27.

## Current status

| Milestone | Result | Report |
|---|---|---|
| Phase 1 · Static game data | Extracted 88 skills, 55 enemy variants, and 11 regions from Dungeon & Rogue 1.00. | [Game mechanics](docs/game_mechanics.md) |
| Phase 2 · Python simulator | Seeded CLI and headless play through floor 100. The independent source audit found two Imitation reward cases, which were fixed and verified. Steam runtime equivalence is still unverified. | [Simulator guide](docs/simulator.md) · [Fidelity audit](docs/simulator_fidelity.md) |
| Phase 2.5 · Local feasibility | Completed the CPU worker sweep and CUDA MLP benchmark. The synthetic pipeline reached 52,841 steps/s at 12 workers in repeated runs. | [Hardware and throughput benchmark](docs/hardware_benchmark.md) |
| Phase 3A · Random baseline | Ran 1,000 games with 16 workers and logged both seeds and episode outcomes in SQLite. No clears, timeouts, or errors. The furthest run reached floor 21. | [Random baseline and plots](docs/baseline_random_1000.md) |
| Phase 3A · Million-game baseline | Completed 1,000,000 episodes with skill and encounter telemetry. Recorded 999,913 defeats, 87 action-limit timeouts, and zero clears or execution errors. | [Results and plots](docs/baseline_random_1m_results.md) · [Runner guide](docs/baseline_random_1m.md) |
| Phase 3A · Timeout investigation | Replayed all 87 timeouts and reproduced the Counter–Mimic combat deadlock with the original Ren’Py combat code. | [Mimic timeout analysis](docs/mimic_timeout_analysis.md) |
| Simulator combat display and replay check | Added combatant and skill types and explicit condition displays. Replayed 1,000 baseline episodes, including 550 that entered a shop, with complete outcome parity. | [Combat display audit](docs/combat_display_audit.md) · [Seeded replay](docs/random_replay_1000.md) |
| Random baseline shop exits | Replayed all 14,120 episodes that reached floor 11 or higher and analyzed 14,142 shop exits by allocation, skill loadout, gold and subsequent floor reach. | [Shop exit analysis and plots](docs/shop_exit_replay/report.md) |
| Phase 3B · Combat and Reward RL | Added a standalone 583-input PPO trainer with two masked heads, uniformly random shop and event actions, checkpoint resume, and paired evaluation. | [Local training and validation](docs/combat_reward_rl.md) |
| Phase 3B · 100,000-episode run | Published the completed portion of the local run. At 100,000 episodes the fixed-seed evaluation averaged floor 10.68 versus 3.51 for paired uniform random. Both policies recorded zero clears on the 128-seed panel. | [Results and plots](docs/combat_rl_100k/training_report.md) |
| Phase 3B-2 · Training performance | Profiled the PPO runtime and added a separate million-episode launcher with batched CPU inference, CUDA updates, and grouped workers. | [Performance results and launcher](docs/combat_reward_performance.md) |
| Phase 3B-2 · Million-episode training run | Completed 1,000,000 training episodes. The final checkpoint averaged floor 11.31 versus 3.51 for paired uniform random on 128 evaluation seeds. Both policies recorded zero clears. | [Results and learning curves](docs/combat_rl_1m/training_report.md) |
| Phase 3C · Alternating Masked PPO and DQN | Replaced randomized shop exits with three ordered, masked DQN heads, milestone targets, and versioned replay. The new run uses a separate output directory. | [Design, verification and launchers](docs/alternating_combat_shop_rl.md) |
| Phase 3C · Archived 200,000-episode DQN experiment | Classified as a failed experiment and excluded from final performance comparisons. The original code, checkpoints and analysis are preserved. | [Interim report, tables and figures](docs/alternating_rl_200k/README.md) |
| Phase 3C · Baseline-relative Shop redesign | Compares legal completed configurations against immediate shop exit. Alternates a fresh Shop learner with the pretrained Combat and Reward PPO. | [Design, verification and Windows launchers](docs/baseline_relative_shop.md) |
| Phase 3C · Baseline-relative Shop at 100,000 episodes | The combined policy averaged floor 11.84 against 12.46 for no-edit shops on the matched 128-seed panel. The combined maximum was floor 33. | [Results, tables and plot](docs/baseline_relative_shop_100k/README.md) |

Greedy and heuristic baselines and Steam integration are upcoming. The hardware benchmark uses synthetic updates, and the 90% target has yet to be evaluated in Steam.

## Run locally

Python 3.10 or newer runs the simulator with the standard library. On Windows, the baseline launcher uses the project virtual environment or the Python 3.12 launcher.

```console
python -m simulator --seed 42
python -m unittest discover -s tests -v
python tests/validate_game_data.py
```

To reproduce the 1,000-game baseline on Windows, create a project `.venv` with Python 3.12, install the [baseline dependencies](experiments/requirements-baseline.txt), and run `run_baseline_random_1000.bat`. See the [baseline report](docs/baseline_random_1000.md#local-records-and-reproduction) for output paths and reproducibility details.

For the resumable experiment, double-click `run_baseline_random_1m.bat`. It defaults to one million episodes and 16 workers. See the [runner guide](docs/baseline_random_1m.md) for the checkpoint and output locations.

## Project approach

- Use the Python simulator for reproducible combat, progression, and reward decisions.
- Store experiment settings, seeds, outcomes, and metrics in SQLite. Analyze them with scikit-learn once that stage begins.
- Train and evaluate policies locally with PyTorch, using baseline strategies and held-out seeds.
- Integrate a local macro with the Steam game, then measure the clear rate across full runs and report uncertainty.

The [observation inventory](docs/observation_space.md) identifies what can be read from the game screen, tracked from play history, or known only to the simulator. [Known issues](docs/known_issues.md) lists source anomalies and remaining verification limits.

## Reporting convention

Write all project reports in English. For each completed milestone, add a short result and a link to the full report in this README's **Current status** table. Keep detailed methods, measurements, tables, and plots in the linked report.

The original game source, game assets, and credentials stay outside this public repository.
