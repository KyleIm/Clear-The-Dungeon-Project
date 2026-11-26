# Clear The Dungeon Project

Clear The Dungeon Project studies decision-making in the roguelike game Dungeon & Rogue. Its goal is to achieve a clear rate of at least 90% in the actual Steam game. That figure is a target for evaluation.

## Approach

- **Simulator.** Build a Python simulator of combat, progression, rewards, and randomness so policies can train against a reproducible game model.
- **Local reinforcement learning.** Train and evaluate policies locally with PyTorch. Compare learned policies with baseline strategies across held-out seeds.
- **Experiment analysis.** Use scikit-learn to analyze experiment data and investigate which decisions and encounter conditions are associated with wins and losses.
- **Experiment records.** Use SQLite to store run configurations, seeds, outcomes, and evaluation metrics.
- **Steam integration.** Develop a local macro interface to observe game state and translate policy decisions into inputs for the actual Steam game. Measure the clear rate across full runs and report the number of runs and uncertainty.

## Roadmap and current status

1. **Repository setup** — initial directory layout and project documentation.
2. **Simulator** — implement the rules and verify transitions against observed game behavior.
3. **Training** — establish baselines and train PyTorch policies locally.
4. **Analysis** — record runs in SQLite and analyze results with scikit-learn.
5. **Steam evaluation** — integrate the macro interface and test whether the real-game clear rate reaches 90%.

The project is at the repository setup stage. Simulator, training, analysis, and macro implementation are planned. The 90% target has yet to be measured.

The original game source, game assets, and credentials are kept outside this public repository.
