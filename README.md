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
| Phase 3B · RL environment design | Specified Steam-observable inputs, masked actions, reward candidates, a Gymnasium adapter, and an independent evaluation protocol. | [RL environment design](docs/rl_environment_design.md) |
| Phase 3C · Local RL implementation | Added a standalone masked PPO learner, nine curriculum stages, resumable checkpoints, paired evaluation and Windows launchers. Validation uses tiny local fixtures. | [Local RL implementation and launch instructions](docs/rl_implementation.md) |
| Phase 3C · Evaluation performance | Diagnosed the first evaluation stall, added CPU workers and CUDA batch inference, and made evaluation progress and time visible. | [Evaluation investigation and benchmark](docs/rl_evaluation_performance.md) |
| Phase 3D · Strategic shop policy | Added a transaction-oriented shop interface, curriculum-controlled repeat costs, paired greedy and seeded stochastic evaluation, and decision statistics. | [Shop redesign and validation](docs/shop_policy_redesign.md) |
| Phase 3D · Completed local run | Finished 1,000,008 steps at Stage 4 with 20,931 completed training episodes. | [Training report](docs/phase3d_1000008/training_report.md) · [Final checkpoint](models/phase3d_1000008.pt) |

Full local training, greedy and heuristic comparisons, and Steam evaluation are upcoming. The 90% target has yet to be evaluated in Steam.

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
