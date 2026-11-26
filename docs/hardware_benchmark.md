# Phase 2.5 Local Hardware and Training Feasibility Assessment

The initial measurements were taken on 2026-11-25 local time, using simulator commit `2cd09698b9940beecc815123f79433aa8b60cd49`. The benchmark ran on a local Windows 11 machine with the existing game logic and dependencies.

Sections 1–4 report the initial hardware and environment measurements. Section 5 reports measurements after CUDA-enabled PyTorch was installed in the project virtual environment and gives updated time estimates.

**For simulation alone, 16 workers achieved the highest throughput at approximately 90,193 steps/s.** The initial assessment suggested 8 workers for a concurrent learner. Eight workers achieved 70,380 steps/s and used about 50% of logical CPU capacity. Sixteen workers used about 94%. Section 5 gives the updated recommendation based on a synthetic learner running alongside the simulator.

## 1. Local Hardware and Runtime

| Item | Result |
|---|---|
| OS and power mode | Windows 11 Home, build 26200. Windows Balanced power mode. |
| CPU | AMD Ryzen AI 7 350 w/ Radeon 860M, 8 physical cores / 16 logical CPUs. |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU. `nvidia-smi` reported 8,151 MiB of total VRAM, with 7,887 MiB free at the time of inspection. The system also detected an integrated AMD Radeon GPU. |
| RAM | 32 GiB installed (16 GiB × 2 at a configured speed of 5600 MT/s). Windows reported 31.12 GiB of visible memory. |
| SSD | 1.024 TB NVMe SSD. The NTFS C: volume hosting the project had 629.81 GiB (676.26 GB) free. |
| Python | CPython 3.12.10, 64-bit Windows AMD64. |
| NVIDIA driver and CUDA | Driver 617.42, WDDM, CUDA UMD 13.4 as reported by `nvidia-smi`, and GPU compute capability 12.0. `nvcc` was absent from PATH. The default CUDA Toolkit install directory showed no installation. |
| PyTorch and GPU computation | `import torch` failed with `ModuleNotFoundError` in the benchmark interpreter. `torch.cuda.is_available()` and CUDA kernel execution were unverified. The process measurements used the standard library and Windows APIs because `psutil` was absent. |

Windows CIM and Storage cmdlets identified the CPU, RAM, SSD, and OS. GPU details came from `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv`, and Python details came from `python -VV`. The CUDA UMD value indicates the driver-reported compatibility level. For a subsequent install, select a CUDA build for Windows x64, Python 3.12, and Blackwell (SM 12.0). Verify the PyTorch version, bundled CUDA version, device name, and successful execution of a tensor operation on the device. PyTorch's Windows Blackwell guidance specifies Windows driver 580.88 or later for CUDA 13.0+ wheels. The measured 617.42 driver meets that requirement. Verify the selected build by running it. [PyTorch Blackwell build guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/) · [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

## 2. Simulator Measurements

### Method

- One environment step is one call to `Game.step(action)` and includes combat, reward, event, and shop actions. A combat turn is a step whose phase immediately before execution is `combat`. Calls to `Game.reset(seed)` after termination were counted separately, and their time was included in overall throughput.
- Each process loaded `GameData` once and ran an independent `Game` with a fixed-seed uniform random policy. The policy selected from `get_legal_actions()`. The benchmark included the full state snapshot returned by `step()`. The 1-worker case ran directly in a single process. The 2/4/8/16-worker cases used Windows `spawn`.
- After a short warm-up, a common start signal began each measurement window of approximately 4 seconds. Each worker configuration ran 3 times. The table reports medians and the step throughput range. Process startup and data loading took place outside the measurement window. Workers sent only aggregate results to the parent process, so the measurement excludes observation serialization, per-step IPC, model inference, and saving.
- CPU use is the sum of worker process CPU time during the measurement window divided by elapsed wall-clock time. The percentage uses 16 logical CPUs as total capacity and excludes other programs. RAM values sum the workers' working sets and private committed bytes reported by Windows `K32GetProcessMemoryInfo`. Summed working sets can count shared pages more than once. Private bytes measure committed memory, regardless of residency.

### Results by Worker Count

| Workers | Median environment steps/s (min–max) | Combat turns/s | Post-termination resets/s | Logical CPU equivalents used / total capacity | Total working set at end | Sum of individual peak working sets | Total private bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11,984 (11,932–15,976) | 11,772 | 66.75 | 1.00 / 6.25% | 23.57 MiB | 24.64 MiB | 16.20 MiB |
| 2 | 28,459 (28,434–28,938) | 28,120 | 115.25 | 2.00 / 12.48% | 45.34 MiB | 48.24 MiB | 31.33 MiB |
| 4 | 49,501 (49,407–50,342) | 45,941 | 1,056.21 | 3.99 / 24.95% | 90.80 MiB | 96.54 MiB | 62.76 MiB |
| 8 | 70,380 (68,951–71,011) | 60,898 | 2,831.64 | 7.93 / 49.58% | 181.77 MiB | 192.37 MiB | 124.11 MiB |
| 16 | 90,193 (88,599–90,530) | 75,293 | 4,341.52 | 15.06 / 94.15% | 361.72 MiB | 383.61 MiB | 248.63 MiB |

The mix of resets and combat turns varied by configuration because the random policy reached different depths. The table gives the resets that occurred in that workload. A separate measurement below gives the throughput limit of `reset()` itself. Across the mixed single-process run, total time spent in combat step calls yielded 12,582 turns/s. The median cost of process creation, data loading, warm-up, and readiness signaling was 0.281/0.301/0.390/0.605 seconds for 2/4/8/16 workers, respectively. Total system RAM use was approximately 17.0 GiB before and after each group of runs. Each process may have reached its peak working set at a different time.

As a control, each configuration ran twice with the same seed and **exactly 16,000 steps per worker**. Median steps/s were 13,501 with 1 worker, 87,941 with 8 workers, and 95,553 with 16 workers. In this control, 16 workers were about 9% faster than 8 workers. In the main experiment with different seeds, they were about 28% faster. Both methods favored 16 workers, with the size of the gain depending on the workload mix. The short windows leave sustained concurrent training and heat-related changes in clock speed unmeasured.

### API Call Costs

The table gives the median wall-clock cost per isolated call across 5 runs. Measurements of `get_state()` and the combat action list used the initial state on floor 1. The shop action list was measured at the lineup screen with all skills acquired. Each `reset()` call used the same seed and excluded data reloading.

| Operation | Median per call | Throughput |
|---|---:|---:|
| `get_state()` | 73.03 µs | 13,693 calls/s |
| `get_legal_actions()` at combat start | 0.895 µs | approximately 1,117,521 calls/s |
| `get_legal_actions()` at the shop lineup screen with all skills acquired | 24.71 µs | 40,473 calls/s |
| `reset()` with full initialization and return of the first observation | 243.64 µs | 4,104 resets/s |

In a separate 10,000-step `cProfile` run, the total cumulative time was 3.059 seconds. Calls to `get_state()` accounted for 2.571 seconds, of which `dataclasses.asdict` accounted for 2.492 seconds. `resolve_turn` took 0.160 seconds, and `get_legal_actions()` took 0.034 seconds. These figures include profiling overhead, and summing parent and child times would double-count some work. The profile served to locate bottlenecks. The main bottleneck in the standalone run was the Python work of repeatedly copying the snapshots of all four characters.

## 3. Small MLP and Combined Throughput

**PyTorch was the missing package required for these measurements.** The initial survey ended before installing packages or the CUDA Toolkit. CPU/GPU forward and backward times, peak GPU VRAM, GPU speedup over the CPU, and combined throughput with the simulator remained unmeasured for all 12 combinations of hidden width 64/128/256/512 and batch size 64/256/1024. The free VRAM reported above describes the idle device.

A follow-up benchmark can use synthetic FP32 inputs and outputs of the same sizes and avoid any game data or reward function in the training signal. After warm-up and GPU synchronization, time the forward pass, backward pass, and optimizer update separately. Record the CPU thread count and the GPU's peak allocated and reserved memory. For the combined measurement, run observation collection, batch transfer, and synthetic updates in the same pipeline. Compare that throughput with standalone measurements at the same worker count.

## 4. Estimated Training Cost and Recommended Setup

The table estimates **wall-clock time for environment step collection alone** using `N / R`, where `R` is the median in the main experiment. It excludes updates, inference, IPC, saving, and sustained performance degradation. Throughput can vary if the actual policy visits the shop and later floors more often.

| Environment steps | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 83.4 s | 35.1 s | 20.2 s | 14.2 s | 11.1 s |
| 10,000,000 | 13 min 54 s | 5 min 51 s | 3 min 22 s | 2 min 22 s | 1 min 51 s |
| 100,000,000 | 2 h 19 min | 58 min 34 s | 33 min 40 s | 23 min 41 s | 18 min 29 s |
| 1,000,000,000 | 23 h 11 min | 9 h 46 min | 5 h 37 min | 3 h 57 min | 3 h 5 min |

If the network runs `U` updates for every `B` new steps, estimate the neural network cost separately as `ceil(N / B) × U × measured update time`. The initial measurements leave this term undetermined. Add collection and training times for sequential execution. For concurrent execution, measure the combined effect of the slowest stage, transfer overhead, and CPU contention. For example, at 1 update per 256 steps, collection takes about 3.64 ms with 8 workers and about 2.84 ms with 16 workers. If the update and batch delivery take longer, adding environment workers may yield only a limited training throughput gain.

**The recommended initial setup is a local Python 3.12 virtual environment, 8 environment workers, and 1 separate learner.** Eight workers match the physical core count and left about half of logical CPU capacity available in this benchmark. For collection alone, 16 workers were the fastest measured configuration. The optimal worker count with the learner was unverified at this point. Retest 4/8/16 workers when PyTorch is available. Memory did not constrain any measured simulator worker count. For runs of 100 million or 1 billion steps, measure the memory and disk costs of observation serialization, batch delivery, replay storage, and checkpointing separately.

## 5. CUDA PyTorch Follow-up Measurements

The earlier sections describe the initial survey, when PyTorch was absent and its measurements were pending. The local follow-up measurements below were taken on 2026-11-26 local time, after creating the project virtual environment. The benchmarks used the existing game logic, policy, and reward function.

- The project `.venv` used PyTorch **2.15.0+cu132** from the official Windows CUDA 13.2 index. The wheel's SHA-256 matched the official index. The bundled CUDA version was **13.2**. The CUDA device was an **NVIDIA GeForce RTX 5070 Laptop GPU**, SM 12.0, with 8,123 MiB of total device memory reported by PyTorch. [Official PyTorch 2.15 release](https://pytorch.org/blog/pytorch-2-15-release-blog/)
- A CUDA tensor [1, 2, 3, 4] produced the expected sum of squares, **30**, and CUDA gradient, **[2, 4, 6, 8]**. A separate CUDA MLP ran backpropagation and an AdamW update, and its parameters changed and remained finite.

### Standalone MLP Benchmark

The architecture was input → H → H → 64, with ReLU, FP32, synthetic inputs and targets, MSE, and AdamW. The CPU used 8 intra-op threads and 1 inter-op thread. After warm-up, the benchmark took the median of 3 measurement windows. Overall ms/step includes zeroing gradients, the forward and backward passes, and the optimizer update. The GPU synchronized every 8 updates. F/B/O gives the median times for forward pass plus loss, backward pass, and optimizer update from 24 separate measurements. CPU timings use wall-clock time, and GPU phase timings use CUDA events. The sum of the phase medians can differ from the overall wall-clock median because of the measurement method and launch overhead. VRAM figures are the peak allocated and reserved memory reported by the PyTorch allocator.

| Input width | Hidden H×2 | Batch size | CPU F/B/O (ms) | CPU ms/step · samples/s | GPU F/B/O (ms) | GPU ms/step · samples/s | GPU peak allocated/reserved (MiB) | GPU speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 64 | 0.083/0.137/0.251 | 0.505 · 126,716 | 0.172/0.316/0.519 | 1.058 · 60,487 | 32.7/34.0 | 0.48× |
| 128 | 128 | 256 | 0.135/0.213/0.199 | 0.633 · 404,443 | 0.184/0.402/0.584 | 1.099 · 233,022 | 33.2/36.0 | 0.58× |
| 128 | 128 | 1024 | 0.293/0.480/0.277 | 1.336 · 766,628 | 0.158/0.292/0.554 | 1.037 · 987,314 | 35.3/38.0 | 1.29× |
| 128 | 128 | 4096 | 0.987/1.748/0.477 | 3.233 · 1,266,962 | 0.156/0.366/0.538 | 1.056 · 3,877,596 | 43.5/58.0 | 3.06× |
| 128 | 256 | 64 | 0.117/0.189/0.318 | 0.630 · 101,606 | 0.164/0.293/0.536 | 1.028 · 62,274 | 33.9/36.0 | 0.61× |
| 128 | 256 | 256 | 0.210/0.364/0.318 | 0.949 · 269,828 | 0.172/0.284/0.527 | 1.042 · 245,669 | 34.6/36.0 | 0.91× |
| 128 | 256 | 1024 | 0.633/1.087/0.426 | 2.245 · 456,172 | 0.212/0.445/0.584 | 1.445 · 708,872 | 37.6/40.0 | 1.55× |
| 128 | 256 | 4096 | 2.785/4.853/0.682 | 8.940 · 458,186 | 0.198/0.300/0.524 | 1.094 · 3,742,564 | 50.4/60.0 | 8.17× |
| 128 | 512 | 64 | 0.193/0.333/0.363 | 1.022 · 62,625 | 0.170/0.274/0.552 | 1.028 · 62,229 | 37.7/40.0 | 0.99× |
| 128 | 512 | 256 | 0.570/1.103/0.494 | 2.123 · 120,567 | 0.178/0.297/0.547 | 1.130 · 226,520 | 39.0/42.0 | 1.88× |
| 128 | 512 | 1024 | 1.966/3.624/0.808 | 7.046 · 145,340 | 0.181/0.388/0.549 | 1.296 · 789,967 | 44.3/62.0 | 5.44× |
| 128 | 512 | 4096 | 7.334/13.777/0.803 | 22.124 · 185,136 | 0.513/0.730/0.465 | 1.753 · 2,337,034 | 65.3/82.0 | 12.62× |
| 256 | 128 | 64 | 0.093/0.145/0.281 | 0.558 · 114,680 | 0.180/0.394/0.580 | 1.141 · 56,116 | 33.0/36.0 | 0.49× |
| 256 | 128 | 256 | 0.141/0.231/0.266 | 0.691 · 370,453 | 0.165/0.371/0.561 | 1.069 · 239,475 | 33.5/36.0 | 0.65× |
| 256 | 128 | 1024 | 0.473/0.719/0.448 | 1.345 · 761,315 | 0.174/0.414/0.596 | 1.111 · 921,523 | 35.9/38.0 | 1.21× |
| 256 | 128 | 4096 | 1.366/2.035/0.613 | 3.988 · 1,027,093 | 0.183/0.397/0.593 | 1.347 · 3,040,928 | 45.7/58.0 | 2.96× |
| 256 | 256 | 64 | 0.131/0.219/0.310 | 0.752 · 85,114 | 0.187/0.358/0.578 | 1.446 · 44,273 | 34.4/36.0 | 0.52× |
| 256 | 256 | 256 | 0.346/0.554/0.431 | 1.302 · 196,643 | 0.180/0.350/0.573 | 1.203 · 212,861 | 35.1/38.0 | 1.08× |
| 256 | 256 | 1024 | 1.060/1.795/0.709 | 3.188 · 321,164 | 0.206/0.375/0.574 | 1.166 · 878,157 | 38.5/40.0 | 2.73× |
| 256 | 256 | 4096 | 4.345/7.119/1.003 | 10.547 · 388,362 | 0.390/0.817/1.010 | 2.048 · 1,999,901 | 52.8/60.0 | 5.15× |
| 256 | 512 | 64 | 0.367/0.560/0.654 | 1.796 · 35,644 | 0.257/0.524/0.791 | 1.888 · 33,893 | 38.7/42.0 | 0.95× |
| 256 | 512 | 256 | 0.859/1.434/0.845 | 3.212 · 79,707 | 0.274/0.526/0.830 | 1.946 · 131,583 | 39.9/42.0 | 1.65× |
| 256 | 512 | 1024 | 2.826/4.854/1.102 | 9.219 · 111,076 | 0.276/0.725/0.906 | 1.930 · 530,590 | 45.5/64.0 | 4.78× |
| 256 | 512 | 4096 | 10.690/20.025/1.300 | 32.976 · 124,210 | 0.577/0.804/0.478 | 1.950 · 2,100,535 | 68.0/84.0 | 16.91× |
| 512 | 128 | 64 | 0.144/0.228/0.403 | 0.822 · 77,899 | 0.321/0.670/0.943 | 1.718 · 37,244 | 33.6/36.0 | 0.48× |
| 512 | 128 | 256 | 0.273/0.410/0.435 | 1.185 · 216,072 | 0.262/0.576/0.797 | 1.584 · 161,608 | 34.1/36.0 | 0.75× |
| 512 | 128 | 1024 | 0.791/1.107/0.576 | 2.470 · 414,650 | 0.509/1.103/1.370 | 2.090 · 490,049 | 37.3/58.0 | 1.18× |
| 512 | 128 | 4096 | 4.120/5.256/1.493 | 11.619 · 352,526 | 0.338/0.759/0.917 | 2.516 · 1,628,224 | 50.1/60.0 | 4.62× |
| 512 | 256 | 64 | 0.272/0.405/0.602 | 1.383 · 46,284 | 0.258/0.502/0.790 | 1.736 · 36,858 | 35.5/38.0 | 0.80× |
| 512 | 256 | 256 | 0.576/0.886/0.658 | 2.205 · 116,114 | 0.285/0.571/0.848 | 1.869 · 136,975 | 36.1/38.0 | 1.18× |
| 512 | 256 | 1024 | 2.276/2.851/1.069 | 5.717 · 179,128 | 0.334/0.771/0.920 | 2.910 · 351,869 | 40.3/60.0 | 1.96× |
| 512 | 256 | 4096 | 6.748/9.880/1.131 | 18.798 · 217,895 | 0.311/0.564/0.875 | 2.278 · 1,798,312 | 57.5/94.0 | 8.25× |
| 512 | 512 | 64 | 0.427/0.631/0.718 | 2.274 · 28,146 | 0.264/0.500/0.760 | 1.703 · 37,588 | 40.8/44.0 | 1.34× |
| 512 | 512 | 256 | 0.963/1.552/0.812 | 3.994 · 64,096 | 0.280/0.525/0.776 | 1.626 · 157,417 | 41.6/44.0 | 2.46× |
| 512 | 512 | 1024 | 3.723/5.781/1.165 | 10.092 · 101,471 | 0.245/0.557/0.873 | 1.929 · 530,789 | 48.0/64.0 | 5.23× |
| 512 | 512 | 4096 | 11.791/19.121/1.072 | 33.276 · 123,092 | 0.751/0.962/0.507 | 2.237 · 1,830,661 | 73.5/118.0 | 14.87× |

The GPU was faster than the CPU in **24 of the 36 cases**. Wall-clock speedup ranged from **0.48× to 16.91×**. The lowest speedup came from 128/128/64 (input/H/batch), and the highest came from 256/512/4096.

Median GPU speedups by batch size were 0.61× for 64, 1.08× for 256, 1.96× for 1024, and 8.17× for 4096. These values use the 8-thread CPU baseline.

### Simulator + Batched Inference + Synthetic Updates

Windows spawn workers randomly selected legal actions from the existing engine. They packed 8 state scalars into 256-dimensional FP32 transport data and queued batches of 256 steps. This transport data served as a benchmark input independent of the training observation design. A single GPU learner ran a CPU-to-GPU transfer, a batched forward pass, a separate forward and backward pass, and one AdamW update before synchronizing. The game actions came from the random policy, and the loss targets came from synthetic random values. Throughput counts steps fully consumed by the learner. Each configuration ran twice for at least 12 seconds per run, excluding worker initialization and GPU warm-up. The second repetition reversed the worker-count order.

| Workers | Median processed steps/s (range) | Updates/s | System / benchmark CPU (%) | Mean / peak GPU (%) | Observed peak system RAM (GiB) | Total worker / peak learner working set (MiB) | Peak device VRAM / Torch allocated (MiB) | Queue receive time (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,076 (8,403–11,749) | 39.4 | 15.9 / 7.2 | 3.1 / 17 | 22.09 | 23 / 1301 | 310 / 35.4 | 88.4 |
| 4 | 35,442 (27,647–43,237) | 138.4 | 39.1 / 27.2 | 6.3 / 10 | 22.14 | 91 / 1301 | 310 / 35.4 | 61.3 |
| 8 | 50,675 (45,533–55,817) | 198.0 | 65.9 / 50.7 | 8.5 / 11 | 22.17 | 183 / 1301 | 310 / 35.4 | 41.1 |
| 12 | 52,841 (47,957–57,725) | 206.4 | 85.1 / 66.3 | 7.7 / 13 | 22.56 | 277 / 1301 | 310 / 35.4 | 28.7 |
| 16 | 54,357 (52,740–55,975) | 212.3 | 88.9 / 73.7 | 7.8 / 11 | 22.57 | 372 / 1301 | 310 / 35.4 | 20.6 |

System CPU measures total Windows busy time. Benchmark CPU sums the worker and learner process CPU times and normalizes them to 16 logical CPUs. GPU utilization and device VRAM came from `nvidia-smi` samples taken every 500 ms. Device VRAM includes the CUDA context. Torch allocated memory is the peak tensor allocation. Worker working sets are summed just before exit and may count shared pages more than once. Learner working set and system RAM are observed peaks sampled at approximately 0.5-second intervals. Queue receive time includes waiting and deserialization.

The measurement windows lasted 12.00–12.03 seconds. The highest median was **54,357 steps/s with 16 workers**. Full 2-second intervals after the first 2 seconds ranged from **51,712 to 60,160 steps/s** in that configuration. A standalone GPU update of the same size (input 256, H=256, batch 256) took **1.203 ms/step**, excluding queue receipt, the additional inference pass, and transfer. Combined throughput includes those operations and counts steps fully consumed by the learner.

An additional 60.0-second continuous run with the recommended 12 workers achieved **59,185 steps/s**. Full 2-second intervals after the first 2 seconds ranged from 52,864 to 66,176 steps/s. System CPU usage was 81.7%, mean GPU usage was 9.4%, observed peak system RAM was 21.99 GiB, observed peak device VRAM was 310 MiB, and peak GPU temperature was 66°C.

### Updated Time Estimates for Synthetic Training

Each estimate divides the environment step count by the combined median for the worker count shown. The benchmark ran 1 batched inference pass and 1 synthetic update per 256 steps. The estimates exclude evaluation, a real RL model and loss, replay, checkpointing, and long-run thermal stability. The combined estimates already include the collection time.

| Environment steps | 1 worker | 4 workers | 8 workers | 12 workers | 16 workers |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 1 min 39 s | 28.2 s | 19.7 s | 18.9 s | 18.4 s |
| 10,000,000 | 16 min 32 s | 4 min 42 s | 3 min 17 s | 3 min 9 s | 3 min 4 s |
| 100,000,000 | 2 h 45 min 24 s | 47 min 2 s | 32 min 53 s | 31 min 32 s | 30 min 40 s |
| 1,000,000,000 | 27 h 34 min 3 s | 7 h 50 min 15 s | 5 h 28 min 54 s | 5 h 15 min 25 s | 5 h 6 min 37 s |

For this synthetic setup, start with **12 workers and 1 GPU learner**. This is the fewest workers among configurations that reached at least 95% of peak throughput. Measure again with the actual model, batch size, and observation delivery method.

In the highest-throughput configuration, mean GPU utilization was 7.8%, and queue receipt and deserialization accounted for 20.6% of wall-clock time. The synthetic pipeline also included frequent synchronization for small batches and state collection and transfer. The measurements leave GPU kernel saturation undetermined and combine CPU interprocess transfer and GPU launch overhead.

The environment used PyTorch without NumPy. The benchmark used PyTorch tensor operations and Python buffers, and execution continued after a NumPy bridge initialization warning.

In the final check, `pip check` in `.venv`, all 58 simulator tests, and static game data validation passed. The virtual environment, official wheel, benchmark scripts, and raw logs remained on the local machine.
