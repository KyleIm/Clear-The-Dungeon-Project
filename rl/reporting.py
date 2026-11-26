"""Bounded experiment logging, host telemetry, and offline reports."""

import ctypes
import json
import math
import os
import shutil
import sqlite3
import subprocess
from collections import Counter
from contextlib import closing
from ctypes import wintypes
from pathlib import Path
from time import monotonic


class EpisodeStats:
    """Summarize episodes with at most 100 floor bins and a few counters."""

    def __init__(self):
        self.floors = Counter()
        self.outcomes = Counter()
        self.episodes = 0
        self.successes = 0
        self.confirmed_deadlocks = 0
        self.reward_sum = 0.0
        self.action_sum = 0

    def add(self, row):
        floor = int(row["floor"])
        if not 1 <= floor <= 100:
            raise ValueError(f"Invalid furthest floor: {floor}")
        self.floors[floor] += 1
        self.outcomes[row["outcome"]] += 1
        self.episodes += 1
        self.successes += bool(row["target_success"])
        self.confirmed_deadlocks += bool(row.get("deadlock_mimic_counter") or row["outcome"] == "deadlock")
        self.reward_sum += float(row["reward"])
        self.action_sum += int(row["actions"])

    def summary(self):
        n = self.episodes
        survival = {}
        reached = n
        median_ranks = ((n - 1) // 2, n // 2)
        median_floors = []
        for floor in range(1, 101):
            survival[floor] = reached / n if n else None
            lower = n - reached
            reached -= self.floors[floor]
            median_floors.extend(floor for rank in median_ranks if lower <= rank < n - reached)
        summary = {
            "episodes": n,
            "outcomes": dict(self.outcomes),
            "target_successes": self.successes,
            "max_floor": max(self.floors, default=None),
            "survival": survival,
        }
        for name, numerator in {
            "clear_rate": self.outcomes["clear"],
            "target_success_rate": self.successes,
            "mean_floor": sum(floor * count for floor, count in self.floors.items()),
            "mean_reward": self.reward_sum,
            "mean_actions": self.action_sum,
            "deadlock_rate": self.outcomes["deadlock"],
            "confirmed_deadlock_rate": self.confirmed_deadlocks,
            "timeout_rate": self.outcomes["timeout"],
        }.items():
            summary[name] = numerator / n if n else None
        summary["median_floor"] = sum(median_floors) / 2 if n else None
        return summary


_TABLES = ("updates", "evaluations", "episodes", "events")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS updates (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, stage INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, stage INTEGER NOT NULL,
    policy TEXT NOT NULL, split TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, stage INTEGER NOT NULL,
    split TEXT NOT NULL, policy TEXT NOT NULL, floor INTEGER NOT NULL,
    outcome TEXT NOT NULL, target_success INTEGER NOT NULL, reward REAL NOT NULL,
    actions INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, stage INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS updates_step ON updates(step);
CREATE INDEX IF NOT EXISTS evaluations_step ON evaluations(step);
CREATE INDEX IF NOT EXISTS episodes_step ON episodes(step);
CREATE INDEX IF NOT EXISTS events_step ON events(step);
"""


class ExperimentStore:
    """Write SQLite rows in bounded transactions with one parent-process writer."""

    def __init__(self, path, batch_size=256):
        self.path = Path(path)
        if batch_size < 1:
            raise ValueError("SQLite batch size must be positive")
        self.batch_size = batch_size
        self.pending = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(_SCHEMA)
        except BaseException:
            self.db.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.flush()
            else:
                self.db.rollback()
        finally:
            self.db.close()

    def _insert(self, table, row, fields=(), values=()):
        columns = ("step", "stage", *fields, "payload")
        placeholders = ",".join("?" for _ in columns)
        payload = json.dumps(row, allow_nan=False, separators=(",", ":"))
        self.db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                        (row["step"], row["stage"], *values, payload))
        self.pending += 1
        if self.pending >= self.batch_size:
            self.flush()

    def update(self, row):
        self._insert("updates", row)

    def evaluation(self, row):
        self._insert("evaluations", row, ("policy", "split"), (row["policy"], row.get("split", "eval")))

    def episode(self, row):
        fields = ("split", "policy", "floor", "outcome", "target_success", "reward", "actions")
        enriched = {"policy": "ppo"} | row
        self._insert("episodes", enriched, fields, tuple(enriched[field] for field in fields))

    def log_event(self, row):
        self._insert("events", row)

    def flush(self):
        self.db.commit()
        self.pending = 0

    def watermarks(self):
        """Capture committed row IDs for exact checkpoint-aligned continuation."""
        self.flush()
        return {table: self.db.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]
                for table in _TABLES}

    def restore_watermarks(self, marks):
        """Remove rows written after the selected checkpoint, including same-step rows."""
        for table in _TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE id > ?", (marks[table],))
        self.flush()

    def discard_after(self, step):
        for table in _TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE step > ?", (step,))
        self.flush()


class _WindowsMemory(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("memory_load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in (
            "total_physical", "available_physical", "total_page", "available_page",
            "total_virtual", "available_virtual", "available_extended_virtual",
        )
    ]


class _WindowsTelemetry:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.previous_ticks = self._ticks()

    def _ticks(self):
        idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
        if not self.kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None

        def ticks(field):
            return field.dwLowDateTime | (field.dwHighDateTime << 32)

        return ticks(idle), ticks(kernel) + ticks(user)

    def sample(self):
        ticks = self._ticks()
        previous = self.previous_ticks
        self.previous_ticks = ticks
        busy = None
        if ticks and previous and ticks[1] > previous[1]:
            busy = 100.0 * (1.0 - (ticks[0] - previous[0]) / (ticks[1] - previous[1]))
        state = _WindowsMemory()
        state.length = ctypes.sizeof(state)
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(state)):
            return busy, None, None
        return busy, state.total_physical - state.available_physical, state.total_physical


class Telemetry:
    """Sample system utilization periodically without polling on every environment step."""

    def __init__(self, min_interval=5.0):
        if min_interval < 0:
            raise ValueError("Telemetry interval must be nonnegative")
        self.min_interval = min_interval
        self.last = float("-inf")
        self.nvidia_smi = shutil.which("nvidia-smi")
        self.windows = _WindowsTelemetry() if os.name == "nt" else None
        if not self.windows:
            try:
                import psutil
            except ImportError:
                psutil = None
            self.psutil = psutil
            if psutil:
                psutil.cpu_percent()
        self.cached = None

    def _nvidia(self):
        if not self.nvidia_smi:
            return None, None, None
        command = [self.nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=2,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            utilization, used, total = result.stdout.splitlines()[0].split(",")
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            return None, None, None

        def to_float(value):
            try:
                parsed = float(value.strip())
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None

        return to_float(utilization), to_float(used), to_float(total)

    def sample(self):
        if monotonic() - self.last < self.min_interval:
            return self.cached.copy()
        cpu = ram_used = ram_total = None
        if self.windows:
            cpu, ram_used, ram_total = self.windows.sample()
        elif self.psutil:
            mem = self.psutil.virtual_memory()
            cpu, ram_used, ram_total = self.psutil.cpu_percent(), mem.total - mem.available, mem.total
        gpu, vram, vram_total = self._nvidia()
        self.last = monotonic()
        self.cached = {"cpu_percent": cpu, "ram_used_mib": ram_used / 2**20 if ram_used is not None else None,
                       "ram_total_mib": ram_total / 2**20 if ram_total is not None else None,
                       "gpu_percent": gpu, "vram_used_mib": vram, "vram_total_mib": vram_total}
        return self.cached.copy()


def _sampled_rows(db, table, limit=4000):
    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    interval = max(1, math.ceil(count / limit))
    rows = db.execute(f"""SELECT payload FROM {table}
                         WHERE id % ? = 0 OR id = (SELECT MIN(id) FROM {table})
                            OR id = (SELECT MAX(id) FROM {table}) ORDER BY id""", (interval,))
    return [json.loads(row[0]) for row in rows]


def _label(row):
    policy = row.get("policy", "ppo")
    name = {"random": "Uniform Random Policy", "ppo": "Masked PPO"}.get(policy, policy)
    return name + " (" + row.get("split", "eval") + ")"


def _plot_series(ax, rows, x_key, y_key, label, divisor=1):
    values = [(row[x_key] / divisor, row[y_key]) for row in rows
              if row.get(x_key) is not None and row.get(y_key) is not None]
    if values:
        ax.plot(*zip(*values), label=label, marker="." if len(values) < 5 else None)


def _finish_figure(plt, fig, destination):
    for ax in fig.axes:
        if ax.lines:
            ax.legend(fontsize="small")
        ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(plt, updates, evaluations, destination):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for column, (dimension, title, divisor) in enumerate([
        ("step", "Environment steps", 1), ("episodes", "Training episodes", 1),
        ("elapsed", "Training wall-clock hours", 3600),
    ]):
        for label in sorted({_label(row) for row in evaluations}):
            rows = [row for row in evaluations if _label(row) == label]
            # Evaluation's own episode count is the sample size. Training progress is attached separately.
            x = "train_episodes" if dimension == "episodes" else dimension
            _plot_series(axes[0, column], rows, x, "clear_rate", label + " clear", divisor)
            _plot_series(axes[0, column], rows, x, "target_success_rate", label + " target", divisor)
            _plot_series(axes[1, column], rows, x, "mean_floor", label + " mean floor", divisor)
        _plot_series(axes[1, column], updates, dimension, "mean_floor", "Training mean floor", divisor)
        axes[0, column].set(xlabel=title, ylabel="Rate", ylim=(-0.025, 1.025))
        axes[1, column].set(xlabel=title, ylabel="Furthest floor")
    _finish_figure(plt, fig, destination)


def _plot_survival(plt, rows, training, destination):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    curves = [("Training", training)] + [(_label(row), row) for row in rows]
    for label, row in curves:
        if row and row.get("episodes") and row.get("survival"):
            series = sorted((int(floor), rate) for floor, rate in row["survival"].items() if rate is not None)
            if series:
                ax.step(*zip(*series), where="post", label=label)
    ax.set(xlabel="Floor", ylabel="Fraction that reached this floor", ylim=(-0.025, 1.025))
    _finish_figure(plt, fig, destination)


def _plot_heads(plt, updates, destination):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for phase in ("combat", "reward", "shop", "event"):
        points = [{"step": row["step"], "count": row.get("phase_steps", {}).get(phase),
                   "entropy": row.get("phase_entropy", {}).get(phase)} for row in updates]
        _plot_series(axes[0, 0], points, "step", "count", phase)
        _plot_series(axes[0, 1], points, "step", "entropy", phase)
    for line in ("policy_loss", "value_loss"):
        _plot_series(axes[1, 0], updates, "step", line, line.replace("_", " ").title())
    _plot_series(axes[1, 1], updates, "step", "throughput", "Training")
    for ax, label in zip(axes.flat, ("Head samples", "Policy entropy", "PPO loss", "Steps/second")):
        ax.set(xlabel="Environment steps", ylabel=label)
    _finish_figure(plt, fig, destination)


def _plot_hardware(plt, updates, destination):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    rows = [{"step": row["step"], **row.get("telemetry", {})} for row in updates]
    for ax, (metric, label) in zip(axes.flat, (
        ("cpu_percent", "System CPU (%)"), ("ram_used_mib", "System RAM (MiB)"),
        ("gpu_percent", "GPU (%)"), ("vram_used_mib", "Device VRAM (MiB)"),
    )):
        _plot_series(ax, rows, "step", metric, label)
        ax.set(xlabel="Environment steps", ylabel=label)
    _finish_figure(plt, fig, destination)


def _wilson(rate, n):
    if rate is None or not n:
        return "—"
    z = 1.959963984540054
    denominator = 1 + z**2 / n
    middle = (rate + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return f"{max(0, middle - half):.2%}–{min(1, middle + half):.2%}"


def _summary_line(label, row):
    def value(key, format_spec):
        result = row.get(key)
        return format(result, format_spec) if result is not None else "—"

    cells = [label, value("episodes", ","), value("clear_rate", ".2%"),
             _wilson(row.get("clear_rate"), row.get("episodes")), value("mean_floor", ".2f"),
             value("median_floor", ".1f"), value("max_floor", "d"), value("target_success_rate", ".2%"),
             value("mean_reward", ".3f"), value("mean_actions", ".1f"), value("deadlock_rate", ".2%"),
             value("confirmed_deadlock_rate", ".2%"), value("timeout_rate", ".2%")]
    return "| " + " | ".join(cells) + " |"


def generate_report(db_path, output_dir):
    """Read compact SQLite metrics and write an English Markdown report and four plots."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        training = EpisodeStats()
        fields = ("floor", "outcome", "target_success", "reward", "actions", "deadlock_mimic_counter")
        for row in db.execute("""SELECT floor, outcome, target_success, reward, actions,
                              json_extract(payload, '$.deadlock_mimic_counter')
                              FROM episodes WHERE split = 'train'"""):
            training.add(dict(zip(fields, row)))
        updates = _sampled_rows(db, "updates")
        evaluations = _sampled_rows(db, "evaluations")
        latest = [json.loads(row[0]) for row in db.execute("""
            SELECT payload FROM evaluations WHERE id IN
            (SELECT MAX(id) FROM evaluations GROUP BY policy, split) ORDER BY policy, split
        """)]
        stages = [json.loads(row[0]) for row in db.execute("""
            SELECT payload FROM evaluations WHERE id IN
            (SELECT MAX(id) FROM evaluations GROUP BY stage, policy, split) ORDER BY stage, policy, split
        """)]

    # Track completed training episodes and elapsed time for old or out-of-band evaluation rows.
    progress = sorted(updates, key=lambda row: row["step"])
    progress_index = 0
    for row in sorted(evaluations, key=lambda row: row["step"]):
        while progress_index + 1 < len(progress) and progress[progress_index + 1]["step"] <= row["step"]:
            progress_index += 1
        if progress and progress[progress_index]["step"] <= row["step"]:
            row.setdefault("train_episodes", progress[progress_index].get("episodes"))
            row.setdefault("elapsed", progress[progress_index].get("elapsed"))

    summary = training.summary()
    _plot_curves(plt, updates, evaluations, output / "learning_curves.png")
    _plot_survival(plt, latest, summary, output / "floor_survival.png")
    _plot_heads(plt, updates, output / "head_statistics.png")
    _plot_hardware(plt, updates, output / "hardware_metrics.png")
    report = output / "training_report.md"
    last = updates[-1] if updates else {}
    progress_text = (f"Recorded training progress reached {last['step']:,} environment steps."
                     if last else "The database contains no training updates.")
    if last.get("elapsed") is not None:
        progress_text += f" Recorded training wall-clock time was {last['elapsed'] / 3600:.3f} hours."
    if last.get("throughput") is not None:
        progress_text += f" The last measured throughput was {last['throughput']:.1f} steps/s."
    report.write_text("\n".join([
        "# Local reinforcement learning results", "", progress_text, "",
        "The table shows all completed training episodes and the latest evaluation for each stage, policy, and split. "
        "Evaluation rows retain their own seed manifest and stopping profile in SQLite when supplied. "
        "Only panels that share seeds and stopping rules support a direct performance comparison. "
        "Training curriculum success is aggregated across the target active in each episode.", "",
        "| Cohort | Episodes | Clear rate | 95% clear interval | Mean floor | Median floor | Max floor | Target success | Mean reward | Mean actions | Early deadlock | Confirmed lock | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _summary_line("Training (all stages)", summary),
        *(_summary_line(f"{_label(row)} · stage {row['stage']} · step {row['step']:,}", row) for row in stages),
        "", "Rates use completed records. A confirmed lock can also count as a timeout in the faithful evaluation profile. "
        "The historical million-game random run had zero clears and used its own seed manifest.", "",
        "## Learning curves", "", "![Learning curves](learning_curves.png)", "",
        "## Floor-wise survival", "", "![Floor-wise survival](floor_survival.png)", "",
        "## Policy heads and updates", "", "![Policy head statistics](head_statistics.png)", "",
        "## Hardware utilization", "", "![Hardware utilization](hardware_metrics.png)", "",
    ]), encoding="utf-8")
    return report
