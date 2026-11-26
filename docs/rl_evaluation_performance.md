# Local RL evaluation performance

## First evaluation at 50,688 steps

The original training database contains a completed eight-episode validation panel at 50,688 environment steps. The run used 12 workers and 128 decisions per worker in each PPO update. Validation started at the first completed update at or above the configured 50,000-step threshold, which was `33 × 12 × 128 = 50,688`. The validation code awaited all eight episodes and printed progress only after the panel returned.

Three episodes reached the unchanged 10,000-action cap on floor 10. The other five ended after 15–39 actions each. Together they took **30,126 decisions** and **52.98 seconds** summed across the sequential episode timers. The old evaluator advanced one game at a time and sent one observation to CUDA for each decision. Its greedy decision path constructed a categorical distribution, computed a value, and read one action back to the CPU. Thirty thousand nearly single-item calls made the three long episodes disproportionately expensive. A later interruption in the run was logged while the old evaluator was inside the CUDA categorical normalization path.

Replaying the saved stage-1 checkpoint with the same eight seeds reproduced every recorded episode's floor, action count and outcome. The last actions revealed two same-slot skill swap cycles, one alternating Meditation with Wolf Fang and another alternating Skull Dagger with Fracture. The third episode repeatedly decreased the HP allocation and then added 10. All three were shop timeouts under the greedy policy. These choices had legal exits available. The separate Mimic–Counter lock detector was false for each. [Mimic timeout investigation](mimic_timeout_analysis.md)

## Evaluation changes

- CPU workers independently advance seeded simulator episodes and refill as episodes finish. One CUDA call scores the active observations per round. Greedy inference uses the existing encoder and policy heads with phase gating and legal masks, and skips distribution normalization and the critic. The model parameters, training path, simulator and rewards are unchanged.
- Progress reports completed episodes, cumulative actions, the longest active episode against the cap, elapsed time and an ETA. ETA is provisional. Shop cycles and late cap hits can shift it substantially as the mix of short and long episodes becomes known.
- SQLite stores per-panel wall time, policy and batching time, worker/IPC time, decision rounds and total actions. The policy timer includes CPU tensor preparation and the synchronous action readback. Worker time includes pipe waits, simulation and resets. The remainder includes process setup and teardown, logging and accounting. Episode timers overlap in parallel mode and must not be added to estimate panel wall time.
- `latest.pt` is saved before each scheduled panel. A resumed run retries a still-pending panel from that checkpoint, with the corresponding SQLite boundary. Training budget accounting remains based on collection and PPO update time. Validation duration has a separate cumulative counter for completed panels.

## Local validation setup

The isolated CUDA run used 12 CPU workers, a 10,000-step budget, a 2,000-step validation interval, eight validation episodes, disabled auto-advance and the normal 10,000-action evaluation cap. The validation copy used 42 decisions per worker per rollout to check closer to the requested interval. It finished at 10,008 training steps because one vector decision advances 12 environments. The original run directory, checkpoints, results and local eight-episode setting were preserved. Raw artifacts stayed local.

The run validated at steps 2,016, 4,032, 6,048 and 8,064. It recorded **15.22 seconds of training** and **70.16 seconds of validation** using the first batched evaluator. The final inference path was measured separately against fixed checkpoints, with no further training. The 8-episode run uses eight active workers at most and the 128-episode run uses all 12.

## Measured runs

| Checkpoint and panel | Episodes | Actions | Timeouts | Panel wall time | Policy and batching | Workers and IPC |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint, historical sequential run | 8 | 30,126 | 3 | 52.98s summed episode time | — | — |
| Original checkpoint, batched replay | 8 | 30,126 | 3 | 20.50s | 12.73s | 6.97s |
| Isolated 10,008-step checkpoint, batched | 8 | 30,115 | 3 | 13.87s | 6.72s | 6.40s |
| Same 10,008-step checkpoint, batched | 128 | 511,981 | 51 | 77.93s | 39.08s | 37.08s |

The matched historical replay took about **2.6× less time by the recorded timers**. The historical number excludes outer evaluation overhead, and the new panel timing includes worker startup and teardown. All 128 episodes completed with an outcome. The 51 capped episodes ended in the floor-10 shop, repeatedly entering or leaving the cost menu. The remaining 77 ended in defeat. No Mimic locks were detected in that panel. The 128-episode panel used 50,075 decision batches, averaging 10.2 actions per batch. The policy path accounted for about 50% of wall time and worker/IPC calls for about 48%. The high action count from shop cycles remains the main workload driver. Evaluation retains the original action cap and reports those outcomes as timeouts.

To reproduce a small local run, copy `rl/training.json`, set `budget.steps` to 10,000, `ppo.rollout_size` to 42, `evaluation.interval_steps` to 2,000, `evaluation.validation_episodes` to 8, and `evaluation.auto_advance` to `false`. Keep 12 workers and CUDA. Use a separate output directory with `run_training.bat --config <copy> --output <new-directory>`. To time a 128-episode panel on the resulting checkpoint, edit only the copied validation count and run `run_evaluation.bat --config <copy> --output <new-directory> --split validation --policy ppo`. This writes a separate evaluation database.
