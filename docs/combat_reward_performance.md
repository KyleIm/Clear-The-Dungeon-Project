# Phase 3B-2 · Combat and Reward PPO performance

The optimization changes how the trainer transports decisions and reuses values. It preserves the 583 inputs, network and two masked heads, PPO update and hyperparameters, +1 floor reward, shop and event random policy, and simulator. The earlier [100,000-episode run](combat_rl_100k/training_report.md) and its local checkpoint and SQLite database are preserved.

## How the runs were measured

Each throughput configuration uses the same master seed, simulator action cap, 12 logical live slots, 128 rollout rounds, 256-transition PPO minibatches and four update epochs. Changing the number of processes only changes how the slots share workers. The smaller inference batch splits the same logical batch into two actor calls. These settings preserve the amount of data requested per rollout. CPU and CUDA use different Torch sampling streams and floating-point kernels, so cross-device policies can take different trajectories despite matching initial seeds. Both episode throughput and actual primitive game-action throughput are reported.

The benchmark performs real PPO updates on small, independent training runs in separate directories. It omits fixed-panel evaluation and report generation to isolate training. Before timing, it warms each learner backend on disposable weights and warms the inference path again for each run. Training time includes collection, PPO, and in-rollout SQLite episode writes. Wall time also includes startup, checkpointing, other database writes and shutdown. Resource figures are sampled across each run, including startup and shutdown. CPU percentage is the process tree normalized to the host’s logical CPU capacity. Peak process-tree RSS sums sampled resident memory and can count shared mappings more than once. NVIDIA utilization and VRAM are device-wide samples.

Phase timings are collected in separate detailed runs with CUDA synchronization around the measured sections. These timings should not be used as production throughput. Worker CPU computation is the maximum timed worker segment per round. The difference between the full worker round and that maximum estimates serialization, IPC, scheduling and synchronization overhead together. The sum across workers measures parallel work and is not added to wall time. Transfer timers include actor input and metric output preparation on the CPU path, plus CUDA rollout transfer wherever the learner uses CUDA. PPO staging covers CPU rollout assembly and GAE. SQLite timing includes episode serialization and database writes, plus commits triggered by checkpoint boundaries.

## Implementation changes

- The next policy call already computes a Critic value for the same observation under the same weights. The optimized path reuses it for the previous transition in that slot. Terminal transitions retain zero continuation. The rollout tail gets a dedicated Critic call before the weights change.
- CPU inference can reuse a copy of the current learner weights. The copy is refreshed after each PPO update. With CUDA learning, the PPO rollout batch still transfers to GPU for the update. This moves per-round inference away from small CUDA transfers while retaining GPU parameter updates.
- A worker can own several independent game slots. One grouped Pipe exchange carries their commands and replies while the parent preserves slot order. Game seeds and each game's engine and policy RNG remain separate. Inference batch size and SQLite batch size are configurable without changing policy or optimizer settings. SQLite retains WAL mode and `synchronous=FULL`.
- CUDA inference can combine action, log probability and value into one readback. The current evaluation path retains its original learner-device inference and worker layout. Benchmark speedups apply to the training path.

## Throughput and resource use

The laptop used an NVIDIA GeForce RTX 5070 Laptop GPU. Each row is one isolated 1,200-episode run. The parent benchmark process retains its CUDA context across variants, so device-wide VRAM in a CPU row can include that context and other processes. Wall time is for the whole short run.

| Runtime variant | Episodes/s | Actions/s | Wall seconds | CPU | GPU | Peak RSS MiB | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda_baseline` | 87.72 | 3,738.9 | 15.21 | 12.9% | 6.7% | 2,514 | 348 |
| `cuda_reuse` | 92.04 | 3,923.3 | 14.00 | 11.7% | 7.6% | 2,515 | 348 |
| `cuda_bundled` | 83.34 | 3,552.3 | 15.41 | 12.4% | 6.9% | 2,516 | 348 |
| `cpu_reuse` | 121.77 | 5,214.6 | 11.01 | 18.4% | 0.3% | 2,516 | 348 |
| `hybrid` | 106.08 | 4,608.2 | 12.30 | 19.8% | 4.5% | 2,522 | 348 |
| `hybrid_6x2` | 128.30 | 5,573.6 | 10.05 | 17.1% | 3.6% | 2,309 | 348 |
| `hybrid_4x3` | 154.05 | 6,692.3 | 8.51 | 14.9% | 2.4% | 2,237 | 348 |
| `hybrid_6x2_batch6` | 145.54 | 6,322.3 | 8.80 | 16.7% | 4.7% | 2,310 | 348 |
| `hybrid_6x2_sqlite1024` | 164.37 | 7,140.4 | 7.87 | 16.6% | 2.4% | 2,311 | 348 |

The confirmation pass repeated the leading hybrid configurations and measured narrower CPU-only worker layouts.

| Runtime variant | Episodes/s | Actions/s | Wall seconds | CPU | GPU | Peak RSS MiB | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_6x2_sqlite1024` | 158.73 | 6,895.4 | 8.62 | 17.0% | 0.9% | 2,291 | 348 |
| `hybrid_4x3` | 150.61 | 6,542.7 | 8.50 | 15.4% | 8.9% | 2,233 | 348 |
| `hybrid_6x2` | 148.68 | 6,458.9 | 8.67 | 16.9% | 3.8% | 2,310 | 348 |
| `cpu_4x3` | 158.61 | 6,792.3 | 8.10 | 14.9% | 0.0% | 2,235 | 348 |
| `cpu_6x2` | 147.20 | 6,303.9 | 8.84 | 15.9% | 0.0% | 2,312 | 348 |

**Selected profile:** six processes with two simulator games per process, CPU inference in batches of up to 12, CUDA PPO updates, Critic reuse and a 1,024-row SQLite commit threshold. It recorded **158.73–164.37 episodes/s** and **6,895.4–7,140.4 actions/s** across two runs. The warmed reference recorded **87.72 episodes/s** and **3,738.9 actions/s**. This is about **1.81–1.87×** on episodes and **1.84–1.91×** on actions. The strongest CPU-only variant was almost tied on episodes/s in the confirmation pass. These short samples do not establish a durable ranking between those two profiles. The SQLite threshold changed within a runtime comparison and its independent contribution was not established.

## Detailed phase measurements

The instrumented panels each processed 600 episodes. Entries are total seconds with call counts in parentheses. Calls measured by a worker timer separate round computation from the residual IPC estimate. Detailed synchronization and host variation affect the timings. Use the warmed throughput panels for runtime selection.

| Phase | CUDA reference | Selected hybrid |
|---|---:|---:|
| CPU simulation, longest group per round | 0.807 (1,093) | 1.121 (1,113) |
| Worker IPC and synchronization estimate | 0.352 (1,093) | 0.384 (1,113) |
| Worker episode resets | 0.913 (466) | 0.956 (454) |
| Actor and current Critic | 0.875 (1,093) | 0.620 (1,113) |
| Dedicated next-state Critic | 0.205 (1,092) | 0.002 (8) |
| Host-to-device and input preparation | 0.247 (2,194) | 0.077 (1,130) |
| Device-to-host and readback | 0.176 (2,194) | 0.020 (1,130) |
| PPO rollout and GAE staging | 0.027 (9) | 0.037 (9) |
| PPO parameter update | 0.872 (9) | 1.278 (9) |
| Actor weight refresh | — | 0.007 (9) |
| SQLite serialization, writes and commits | 0.042 (611) | 0.043 (611) |
| Checkpoint snapshot and write | 0.152 (2) | 0.167 (2) |

Dedicated Critic calls fell from 1,092 to 8 in these short panels. The baseline spent comparable time in actor inference, PPO updates, worker resets and simulation. Small inference transfers contribute additional latency on each round. SQLite occupied a small share of the measured time. Grouping reduces messages per round, while more than one game per worker can increase the longest local compute interval. Those costs explain why process count, batch splitting and commit interval should be measured together. Timing from a full million-episode run remains unmeasured, and the action count per episode can grow as learning progresses.

## Reproduce locally

Install the project dependencies into `.venv`, including `psutil` for CPU and RAM sampling. An NVIDIA driver with `nvidia-smi` provides GPU utilization and VRAM samples. Start each command with a new output directory.

```console
python -m combat_rl.benchmark data/combat_rl_perf_warm --episodes 1200 --variants cuda_baseline cuda_reuse cuda_bundled cpu_reuse hybrid hybrid_6x2 hybrid_4x3 hybrid_6x2_batch6 hybrid_6x2_sqlite1024
python -m combat_rl.benchmark data/combat_rl_perf_detailed --episodes 600 --detailed --variants cuda_baseline cuda_bundled hybrid hybrid_4x3
python -m combat_rl.benchmark data/combat_rl_perf_confirm --episodes 1200 --variants hybrid_6x2_sqlite1024 hybrid_4x3 hybrid_6x2 cpu_4x3 cpu_6x2
python -m combat_rl.benchmark data/combat_rl_perf_selected --episodes 600 --detailed --variants hybrid_6x2_sqlite1024
```

## Start a fresh million-episode run

Run `run_combat_rl_optimized.bat`. It uses `combat_rl/training_optimized.json` and starts from newly initialized weights in `data/combat_reward_rl_optimized`. Resume that run with `resume_combat_rl.bat --config combat_rl\training_optimized.json`. The original launcher and configuration still target `data/combat_reward_rl`.

## Validation

All 86 tests passed. A matched CPU fixture compared optimized and reference runtime counters and PPO weights within floating-point tolerance. Additional tests checked grouped worker state restore, masked sampling, Critic boundary handling, CUDA updates and the original uniform-random shop replay. The optimized Windows launcher completed four capped training episodes and two evaluation episodes for each policy in a separate directory. The existing 100,000-episode SQLite database and checkpoint had matching SHA-256 hashes before and after the work. The million-episode run was left for local execution.
