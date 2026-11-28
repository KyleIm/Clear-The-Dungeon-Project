# Phase 3C V2 · Adaptive Shop baseline and Reward experience transfer

V2 starts at zero episodes using the final Phase 3B-2 million-episode Combat and Reward PPO checkpoint. Both older Phase 3C runs are preliminary and are excluded from final comparisons. Their repository history, local checkpoints and SQLite files are preserved. V2 starts with fresh Shop and Reward-value models in `data/adaptive_shop_v2`. [Phase 3B-2 benchmark](combat_rl_1m/training_report.md) · [Preliminary DQN run](alternating_rl_200k/README.md) · [Preliminary V1 run](baseline_relative_shop_100k/README.md)

## Schedule and boundary order

| Training episodes | Combat and Reward | Shop |
|---|---|---|
| 1–25,000 | Inherited masked PPO | Uniform random |
| 25,001–50,000 and subsequent even blocks | Frozen PPO. At each Reward decision, sample from PPO with probability 0.8 and uniformly from legal choices with probability 0.2 | Live incumbent frozen. Learn a separate candidate from paired treatments |
| Other alternating blocks through 1,000,000 | Inherited masked PPO, sampling both heads | Most recently accepted incumbent frozen |
| 1,000,001–1,025,000 | Final on-policy PPO adaptation | Incumbent frozen |

The inherited PPO uses its 583 observations, 7 Combat logits, 5 Reward logits, existing hyperparameters and +1 reward per floor. The forced mixture applies only to Reward decisions in even blocks. Only ordinary sampled PPO rollouts enter the PPO update. Game rules and the simulator are unchanged.

At the end of each Shop block, save the pre-boundary checkpoint, evaluate the candidate and incumbent with the same frozen PPO, and accept or reject the candidate. Evaluate the accepted Shop policy with the pre-transfer PPO. Fit Reward Q and transfer its guidance to the Reward output layer, then repeat that evaluation on the same seeds with the accepted Shop policy. Store both results and archive the post-boundary checkpoint. An interrupted boundary is replayed from the last committed checkpoint watermark. The planned 1,025,000-episode comparison uses the **post-transfer** 1,000,000-episode incumbent panel as the start of final adaptation.

## Shop candidate and incumbent

Every visited training shop in an even block generates four continuations. One applies the full frozen incumbent. Each other continuation substitutes a random legal completed configuration for exactly one of Gold, Stats or Skills, while following that incumbent at the other two stages. All stages run Gold → Stats → Skills. Later stages see the purchases, skill costs, stat changes and remaining resources produced by earlier stages. Policy versions and a seed namespace determined by episode policy seed, floor and head make proposals reproducible. The checkpoint also stores the generator version, candidate limit, scoring/tie convention, incumbent weights, and training and RNG state.

Four branches start from the same mutable game and simulator RNG state. They use aligned draws for PPO/mixture, events and actions when their trajectories still permit the same random calls. Each continues to defeat, the next shop or the existing floor-100 or clear milestones after the late shops. The comparison cap right-censors the branch. A resolved label is the sign of the furthest-floor difference, with a clear taking precedence over a floor-100 defeat. Raw floor differences are retained. Shop labels never enter the PPO return.

The generator retains the V1 bounded catalog of deterministic purchases and complete legal configurations. It avoids inverse editing loops. Random Reroll remains in the uniform policy. A scored candidate does not have access to a future Reroll result. The per-stage model retains the 1,370-feature entry and completed-configuration pair, shared 128/64 encoder and three heads predicting loss/tie/win. Score equals predicted win probability minus predicted loss probability. At the first baseline, no edits have known relative value zero. With an edited incumbent, a no-op can have different value and the learner scores it. The new candidate resets each even block and trains on eligible current-PPO comparisons with class-balanced batches. The live incumbent remains separate from those updates. The observations, sparse labels and class balancing can yield uncalibrated scores.

Promotion requires a positive one-sided paired-bootstrap lower bound on both the fixed 128-seed panel and a fresh 256-seed validation panel. Validation seeds are namespaced by boundary and held out from training. The bound uses the 0.25th percentile of 10,000 resamples, approximately a 0.05/20 allowance across the scheduled Shop boundaries on validation. A candidate must also avoid increasing timeout counts or reducing clears in either panel. The utility gives a clear one unit above a floor-100 defeat. This is a conservative approximate test, especially when paired outcomes are sparse. The no-edit policy is evaluated permanently as a separate reference. The single-head diagnostic substitutes one candidate head into the incumbent stack.

## Reward experience transfer

Each live Reward decision in a Shop block records the 583 float32 observations, legal mask, selected reward, branch indicator, PPO probability and marginal behavior probability `0.8 * PPO + 0.2 / legal_count`. It retains HP, current and base stats, floor, equipped skills and uses, gold and the Shop and PPO versions. At episode completion it adds the observed furthest floor, outcome and clear flag. Timeouts remain in diagnostics and are excluded from Q fitting and guidance. The learner keeps sampled windows in RAM and writes the full experience to SQLite.

At a boundary, sample up to 32,768 recent resolved decisions from the just-finished block and its single frozen incumbent version. Stratify by choice source and reward, and hold out a disjoint subset by game seed. A fresh 583→128→5 tanh network estimates the chosen action's final floor minus current floor by minimizing mean squared error on returns scaled by 10. It uses Adam at 0.001 for three passes. This modest model and bounded window limit training cost. Estimates describe return under the behavior policy in that Shop block, can extrapolate poorly, and do not estimate a causal benefit for a rare action in an arbitrary state. The report records held-out error and the Q distribution for each legal choice.

Transfer uses only the frozen encoder's outputs. A softmax of estimated returns at a two-floor temperature proposes a target over supported legal choices. An action needs at least 32 forced observations in the fitting sample to enter this target. Mix the target at 10% into the old masked PPO distribution. Use 2.5% following Shop promotion to account for the change in downstream continuation policy. A four-step Reward-only Adam fit is backtracked to stay below empirical mean KL 0.005 on the held-out states. Reset only the affected Reward optimizer moments before PPO resumes. Combat, encoder and critic weights are unchanged by transfer. This prevents drift of the Combat policy on fixed observations. The subsequent trajectory can still change as Reward choices change.

## Reports and diagnostics

- Fixed panels report floor depth, survival, clears and timeouts against the Phase 3B-2 reference, permanent no-edit and the incumbent. Boundary output identifies accepted policies, single-head interventions and pre/post-transfer performance.
- The report displays **unused stats and mana at every checkpoint** and links exact distributions by policy and floor. It also includes loadouts, skill uses, costs, spending, purchase frequencies and no-edit rates. Small separate seed panels test legal one-point or one-use changes with matched simulator continuations and compare equal-cost alternate allocations when available. Their Q or Shop scores and outcomes are reported separately.
- Reward tables distinguish PPO and forced-random actions and show action probabilities by HP, stat drift and shop distance. The summary exposes high-HP Potion rates and the Gold change on held-out states plus the observed fixed panels. The direction may vary by fixture or training interval.
- Timeouts record the floor, cause, branch, observed player and enemy state and an up-to-24-turn trace. Pattern matches are signatures of a recorded suffix. A Mimic/Sacrifice/Counter match can be compatible with the known loop without proving identical hidden state. Raw timeout details and experience remain local.
- Live and branch throughput, worker timings, Q and transfer time, promotion time, CPU/GPU/RAM samples and SQLite allocated size are reported. The production run uses 12 slots on six Windows workers, CPU batched Combat/Reward inference, inherited critic reuse and CUDA updates. Workers batch the four paired continuation states locally. Complete Shop configurations eliminate repeated decision round trips. The full data file can be large because Reward observations are recorded at every eligible decision.

## Local launchers

- [Start V2](../run_adaptive_shop_rl.bat) uses [the V2 settings](../adaptive_shop/training.json) and stops automatically after the **100,000-episode** boundary has completed.
- [Resume V2](../resume_adaptive_shop_rl.bat) continues the same checkpoint to 1,025,000 by default. Use `--stop-at` to choose an earlier whole boundary.
- [Generate V2 report](../report_adaptive_shop_rl.bat) reads the committed SQLite database and writes `report/report.md` plus aggregate tables and plot in the experiment directory.
- [Validate V2](../validate_adaptive_shop_rl.bat) creates a small separate run. It accepts `--device cpu` when CUDA is unavailable.

The training launchers accept `--config`, `--output`, and `--initial`. Resume requires the same configuration and matching database. The default output directory is separate from the preliminary runs.

## Local validation

The CUDA smoke used 48 training episodes in three 16-game blocks, four slots, four seeds per main and validation panel, and a separate resource-probe panel. It stopped at the first full Shop boundary and resumed into the final PPO-only segment. The run recorded 136 Reward decisions, with 103 from PPO and 33 from forced exploration. In the guidance fixture seven forced Gold examples entered fitting, and the Reward layer changed with a held-out KL of approximately 0.004 while all other PPO tensors stayed identical. The held-out mean Gold probability moved by roughly +0.001 percentage point, a small policy change in this fixture. Both promotion panels tied and kept the incumbent. Tests separately cover active promotion, rejection, the floor-100 clear distinction, and passing changed skill costs from Gold to later heads.

One smoke run measured roughly 27.5 episodes/s and 1,057 live actions/s during its Shop block. The 635 comparison actions took about 0.30 worker-seconds. These timing and return estimates are for a small fixture. The 100,000-episode run is left to the local launcher.
