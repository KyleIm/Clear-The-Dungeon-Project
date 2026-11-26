# Resumable Uniform Random Policy Baseline

The Windows launcher defaults to **1,000,000 episodes with 16 CPU workers**. The published validation completed **1,000 episodes**. The full run is reserved for manual launch.

## Run or resume on Windows

Double-click `run_baseline_random_1m.bat` at the repository root. It changes to the repository directory, uses the project `.venv` when present, and keeps the window open after a double-click run. Install [the baseline plotting dependency](../experiments/requirements-baseline.txt) into that environment if needed. The default run directory is `data/baseline_random_1m/`.

To resume, press Ctrl+C and relaunch with the same output and experiment parameters. The runner commits one SQLite transaction per batch and starts with the next unfinished batch. Episodes in an uncommitted batch may be replayed. A directory lock prevents two writers from running concurrently. The runner checks the fixed seed plan, settings, simulator, data, and experiment code before resuming. Worker count can vary without altering episode seeds. Use a different output directory for a new run. If reporting was interrupted after the simulation finished, relaunching regenerates the report from the committed database.

A smaller run uses a separate directory:

```bat
run_baseline_random_1m.bat --games 1000 --workers 16 --batch-size 32 --output data\baseline_random_1m_validation --max-batches 16
run_baseline_random_1m.bat --games 1000 --workers 16 --batch-size 32 --output data\baseline_random_1m_validation
```

The optional batch limit provides a clean stop after a specified number of new batches. The display reports committed episodes, session elapsed time, throughput, and an ETA based on that session's throughput. SQLite is the source of truth after shutdown.

## Recorded data

| SQLite table | Granularity and fields |
|---|---|
| `run`, `batches` | Settings and input fingerprints, plus the ID and episode count of each committed batch. |
| `episodes` | One row per game with independent deterministic game and policy seeds, initial loadout, highest reached floor, outcome, actions, and elapsed time. |
| `episode_skills` | One row per relevant game/skill. The first acquisition floor is 1 for initial skills and empty for temporary or unlearned casts. Includes equip event count, equipped flag at episode end, sum of observed positive use-count changes, peak and final available uses, and actual casts. |
| `episode_monsters` | One row per game/monster variant with encounter, win, loss, and unresolved counts. An unfinished fight after a timeout or an error is unresolved. |

Use counts are sampled at decision boundaries. Positive increases include allocation, refresh, or newly equipped uses visible at those boundaries. Actual casts count player calls entering the existing skill resolver, including misses, Sacrifice, and a Mimic copy that executes. A selected move blocked before resolution does not count. The local trace probe forwards the original engine call and does not change RNG draws or game rules.

At completion the runner writes `report.md`, `summary.json`, `floor_survival.png`, and `skill_cohorts.png` in the run directory. The report includes per-floor reach and progression, per-monster wins and losses, and conditional floor-5 and floor-10 reach among episodes that acquired, equipped, or cast each skill. JSON also includes clear counts by cohort and acquisition-floor frequency. These are descriptive cohorts. Later skills can be acquired only after surviving earlier floors.

## Storage and validation

The parent process writes SQLite in 128-game batches by default. It keeps at most twice the worker count in flight, and workers aggregate telemetry within each episode and batch. It persists no per-action state snapshots. SQLite uses an 8 MiB page cache, a WAL with durable batch transactions, and disk-backed temporary query storage. Final analysis queries load only aggregate rows into Python.

The Windows launcher validation used 32-game batches and 16 workers. The first invocation committed 512 episodes and the second finished the remaining 488. The first 512 episode IDs and runtime values matched after the resume. SQLite integrity and foreign-key checks passed. Both seed columns together contained 2,000 distinct values.

| Validation measurement | Result |
|---|---:|
| Episodes / completed batches | 1,000 / 32 |
| Defeats / clears / timeouts / errors | 1,000 / 0 / 0 / 0 |
| Highest floor reached | 16 |
| Skill summary rows / actual casts | 6,540 / 12,891 |
| Encounters / wins / losses / unresolved | 3,536 / 2,536 / 1,000 / 0 |
| SQLite file after checkpoint | 348,160 bytes |

A straight extrapolation from this sample gives about 332 MiB for the main database at one million episodes with a similar distribution. Actual disk use depends on encounter depth and can be higher with a WAL and simultaneous report queries. The experiment directory should have free space for both the database and temporary analysis. The seed derivation differs from the original [1,000-game baseline](baseline_random_1000.md), so this validation is a separate sample.
