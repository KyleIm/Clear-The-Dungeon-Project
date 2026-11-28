# Phase 3C V3 · Final long-run experiment

V3 is a fresh run initialized from the Phase 3B-2 Combat and Reward PPO at 1,000,000 episodes. A new Shop model and Reward Q estimator start with fresh weights. V3 writes to `data/adaptive_shop_v3`. Final comparison tables use the original random and Phase 3B-2 PPO references. The prior Phase 3C reports and local artifacts are retained as preliminary experiments. [V2 mechanics and schema](adaptive_shop_v2.md) · [Uniform Random reference](baseline_random_1m_results.md) · [Phase 3B-2 reference](combat_rl_1m/training_report.md)

## Schedule and Reward transfer

The first 25,000 episodes train the inherited PPO with uniform shops. Through 1,000,000, alternate 25,000-episode blocks of Shop learning and PPO learning. Shop blocks freeze the PPO and use the existing 20% uniform exploration at each Reward decision. Combat uses the frozen masked PPO during these blocks. Each Shop boundary saves before evaluating and promoting candidates, records an accepted-policy panel before and after Reward transfer, and saves the committed post-boundary state. Episodes 1,000,001–1,025,000 run on-policy PPO with the accepted Shop stack fixed. The final checkpoint records the same fixed evaluation panel used at 1,000,000 after Reward transfer. The fresh run defaults to the complete budget.

| Reward transfer setting | V2 | V3 |
|---|---:|---:|
| Guidance mixture with incumbent retained | 0.10 | 0.50 |
| Guidance mixture after any Shop promotion | 0.025 | 0.10 |
| Auxiliary Adam updates | 4 | 40 |
| Held-out mean KL limit | 0.005 | 0.05 |
| Q softmax temperature, floors | 2 | 2 |

The 583→128→5 Q model and its observed floor-return target, MSE fitting method and sampling settings are inherited. PPO observations, masks, floor reward and hyperparameters are inherited. Auxiliary training uses the detached shared encoder and updates only the Reward output layer. The KL backtracking records the accepted step fraction and empirical TV distance. Stronger guidance permits more policy movement under observed states. Held-out bounds do not guarantee progress on a different state distribution. Reports retain pre/post masked probabilities, observed choices by HP, stat drift and shop distance, and Q quantiles for those contexts.

## Global and Local search

Each training shop still produces exactly four conceptual continuations. The unchanged stack is the incumbent. Each of the other arms changes only one head. Gold precedes Stats, which precedes Skills, so each arm passes its actual changes to the next decision. Each treated head flips an independent fair coin for Global or Local. Global uses the previous generator of random legal completed configurations. Local starts at the exact incumbent configuration for that head and current preceding state. Identical states or a lack of legal changes can produce ties. Resolved outcomes still use −1, 0 and +1 relative to the incumbent, with a final clear outranking a floor-100 defeat. No resource-use bonus is added.

| Head | Local proposal |
|---|---|
| Gold | One legal deterministic purchase, skill-cost reduction or eligible type change, sampled uniformly from eligible operations. |
| Stats | One legal +1, −1, +10 or −10. After a decrease, a fair coin can move the same amount to a different stat if legal. This allows spending, leaving points unused or changing the allocation at equal total. |
| Skills | Uniformly choose among available operation types, then choose a legal move in that type. Replace a single equipped skill or change one to three uses in one direction. After Forget, a fair coin can add one use to a different affordable slot. |

All edits use the original shop legality checks. Only the real selected configuration changes the live simulator. Executable policy ranking includes the incumbent anchor and up to seven other proposal attempts. Each attempt flips its own fair Global/Local coin. Duplicates and unavailable moves can reduce the number of distinct options. The anchor scores zero by definition and wins ties. This preserves the limit of eight configurations scored by any one head ranker. Previous accepted rankers can run to resolve the anchor. They generate proposals and scores without running game continuations. Equivalent head/prefix work within a shop comparison and deterministic Global slots across immutable rankers are cached.

## Head-wise promotion

Each accepted head snapshots its own weights **and the shared encoder it was trained with**. The other accepted heads retain their exact encoder and generator. The saved recipe includes its candidate limit, generator version and recursive anchor policy. A stage can reconstruct that recipe from a new shop state and the episode policy seed. There can be at most 20 accepted ancestry levels per head over the scheduled Shop boundaries, so later promotions may take longer to plan. Each block fits the same three supervised heads in a new shared candidate network as in V2.

A replacement is measured against the incumbent with identical frozen PPO and paired seeds. The fixed panel has 128 games and the fresh per-boundary validation panel has 256. V3 measures the full candidate and the three single substitutions. Multiple eligible single substitutions are tested together when their subset differs from the full replacement. The measured full combination is ineligible if it would replace a head whose individual intervention shows a negative mean utility on either panel. When several stacks clear the gates, select the one with the strongest development lower bound. A combination that fails its own gates leaves eligible single replacements available.

The lower bound uses the lower 0.05th percentile of the inherited 10,000 paired-bootstrap replicates. This allocates 0.05 across up to 20 boundaries × 5 alternatives. Each panel must have a positive lower bound, at least the configured minimum paired sample, no reduction in clears and no increase in timeouts. The bootstrap is approximate and the fixed panel is reused. The independent validation panel is seeded separately at each boundary. Conservative gates can leave the incumbent unchanged for many blocks. No-edit remains a permanent evaluation arm.

## Checkpoint recovery and resource use

The existing V2 data schema stores observations and compact treatment outcomes on disk with trajectory-sized and bounded fitting buffers in RAM. V3 keeps 12 games across six worker processes, CPU batched PPO inference, critic reuse, local paired simulation and CUDA updates. Only one live run owns the SQLite directory. Periodic saves and every 25,000-episode checkpoint store model weights, optimizer state, worker snapshots, RNGs and database watermarks. Resume rolls back newer SQLite rows beyond the selected latest checkpoint. Use the original settings and a matching SQLite file.

The runner prints measured free space and database/WAL size and records both at boundaries. It checks a configurable 2 GiB reserve before training and after rollouts. If the reserve is crossed during a rollout, it attempts to save before exiting with an error status. A drive failure or a single oversized write can still prevent that last save. The prior approximately 3.1 GiB per 100,000 V2 episodes gives an order-of-magnitude estimate of about 32 GiB for episode data at the full budget. Longer episodes, WAL files, checkpoint archives and reports can raise actual use. The local validation volume had about 608 GiB free. No full-length memory or storage profile was run.

The report adds Global/Local label distributions and offer/selection rates, head-specific promotion reasons, paired combination results, context-stratified Q values and observed Combat skill usage. It retains unused stat and mana distributions, actual allocation probes, Reward probabilities, timeouts, throughput and the paired historical PPO reference. Training outputs, raw experience, checkpoints and databases stay local.

## Windows launchers

- [Start the full run](../run_adaptive_shop_v3_rl.bat) · defaults to **1,025,000 episodes** and generates the report automatically.
- [Resume](../resume_adaptive_shop_v3_rl.bat) · continues the same directory from `latest.pt`.
- [Generate an interim or final report](../report_adaptive_shop_v3_rl.bat) · reads committed data and writes `report/report.md` and its CSVs and plot.
- [Run small validation](../validate_adaptive_shop_v3_rl.bat) · uses a separate local directory.

The launchers use the project virtual environment. [Settings](../adaptive_shop_v3/training.json). The training and resume launchers accept `--config`, `--output` and `--stop-at` for an explicit whole-boundary exit. A default run has no confirmation prompts or 100,000-episode stop.

## Minimum verification

The 48-episode CUDA integration fixture used three shortened blocks and the actual Phase 3B-2 checkpoint. It recorded eight Global and seven Local comparisons. Reward transfer changed only the Reward output tensors, with measured KL 0.01185 and mean TV 0.01500. Before the final adaptation segment, the fixture inserted a committed SQLite row beyond the latest checkpoint watermark and verified that resume removed it. Final adaptation left Shop weights unchanged. The small panels kept the incumbent. Tests separately exercise selective promotion, a regressing head and a losing combination, checkpointed head reproducibility, incumbent-anchored Local search, the eight-option limit and the save-on-low-space path.

The Shop block in that fixture processed 26.2 episodes/s and 932 live actions/s. Its 311 comparison actions took 0.28 worker-seconds. At the final boundary the database and WAL occupied 1.92 MiB. Samples in the resumed session recorded a peak of 3,144 MiB process-tree RSS and 318 MiB GPU memory. On one funded floor-10 state, 20 planning repetitions yielded a median of 6.20 ms for a single saved version per head and 25.55 ms with four versions per head. These local measurements provide a size and latency check. They do not establish full-run throughput or learned strategy quality.
