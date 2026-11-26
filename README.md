# Clear The Dungeon Project

Clear The Dungeon Project studies decision-making in the roguelike game Dungeon & Rogue. Its goal is to achieve a clear rate of at least 90% in the actual Steam game. That figure is a target for evaluation.

## Approach

- **Simulator.** Build a Python simulator of combat, progression, rewards, and randomness so policies can train against a reproducible game model.
- **Local reinforcement learning.** Train and evaluate policies locally with PyTorch. Compare learned policies with baseline strategies across held-out seeds.
- **Experiment analysis.** Use scikit-learn to analyze experiment data and investigate which decisions and encounter conditions are associated with wins and losses.
- **Experiment records.** Use SQLite to store run configurations, seeds, outcomes, and evaluation metrics.
- **Steam integration.** Develop a local macro interface to observe game state and translate policy decisions into inputs for the actual Steam game. Measure the clear rate across full runs and report the number of runs and uncertainty.

## Roadmap and current status

1. **Repository setup** — complete.
2. **Phase 1 — Static game data and mechanics** — complete. The baseline covers 88 skills, 55 monster variants, and 11 regions.
3. **Simulator** — implement the rules and verify transitions against observed game behavior.
4. **Training** — establish baselines and train PyTorch policies locally.
5. **Analysis** — record runs in SQLite and analyze results with scikit-learn.
6. **Steam evaluation** — integrate the macro interface and test whether the real-game clear rate reaches 90%.

The current milestone is a static JSON baseline for Dungeon & Rogue 1.00. Simulator, training, experiment storage, and macro integration are planned. The 90% Steam clear-rate target has yet to be measured.

See the [game mechanics](docs/game_mechanics.md) and [known issues](docs/known_issues.md). Validate the datasets with `python tests/validate_game_data.py` using the Python standard library.

The original game source, game assets, and credentials are kept outside this public repository.
