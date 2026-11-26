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

from .decision_stats import REWARDS


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
CREATE TABLE IF NOT EXISTS decision_stats (
    evaluation_id INTEGER PRIMARY KEY REFERENCES evaluations(id) ON DELETE CASCADE,
    payload TEXT NOT NULL
);
CREATE VIEW IF NOT EXISTS evaluations_with_decisions AS
SELECT e.id, e.step, e.stage, e.policy, e.split,
    CASE WHEN d.evaluation_id IS NOT NULL
         THEN json_set(e.payload, '$.decision_stats', json(d.payload))
         ELSE e.payload END AS payload
FROM evaluations e LEFT JOIN decision_stats d ON d.evaluation_id = e.id;
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
            self.db.execute("PRAGMA foreign_keys=ON")
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

    def _insert(self, table, row, fields=(), values=(), *, decisions=None):
        columns = ("step", "stage", *fields, "payload")
        placeholders = ",".join("?" for _ in columns)
        payload = json.dumps(row, allow_nan=False, separators=(",", ":"))
        decision_payload = (json.dumps(decisions, allow_nan=False, separators=(",", ":"))
                            if decisions is not None else None)
        row_id = self.db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                                 (row["step"], row["stage"], *values, payload)).lastrowid
        if decision_payload is not None:
            self.db.execute("INSERT INTO decision_stats VALUES (?, ?)", (row_id, decision_payload))
        self.pending += 1
        if self.pending >= self.batch_size:
            self.flush()

    def update(self, row):
        self._insert("updates", row)

    def evaluation(self, row):
        summary = dict(row)
        decisions = summary.pop("decision_stats", None)
        self._insert("evaluations", summary, ("policy", "split"),
                     (summary["policy"], summary.get("split", "eval")), decisions=decisions)

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


def _sampled_rows(db, table, limit=4000, project=None):
    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    interval = max(1, math.ceil(count / limit))
    rows = db.execute(f"""SELECT payload FROM {table}
                         WHERE id % ? = 0 OR id = (SELECT MIN(id) FROM {table})
                            OR id = (SELECT MAX(id) FROM {table}) ORDER BY id""", (interval,))
    values = (json.loads(row[0]) for row in rows)
    return [project(row) if project else row for row in values]


def _label(row):
    policy = row.get("policy", "ppo")
    name = {"random": "Uniform Random Policy", "ppo": "Masked PPO",
            "ppo_deterministic": "Masked PPO deterministic",
            "ppo_stochastic": "Masked PPO stochastic"}.get(policy, policy)
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


def _decision_trend_projection(row):
    decisions = row.pop("decision_stats", None)
    if decisions is not None:
        row["reward_rates"] = decisions["reward"]["rates"]
        shop = decisions["shop"]
        row["shop_rates"] = {key: shop.get(key) for key in (
            "exit_rate", "repeat_rate", "episode_shop_timeout_rate", "shop_reach_rate",
            "floor_10_reach_rate", "mean_leave_probability",
        )}
    return row


def _plot_decision_trends(plt, evaluations, destination):
    labels = sorted({_label(row) for row in evaluations if "reward_rates" in row})
    fig, axes = plt.subplots(len(labels), 2, squeeze=False, figsize=(13, 3.6 * len(labels)), layout="constrained")
    for label, (reward_ax, shop_ax) in zip(labels, axes):
        rows = [row for row in evaluations if _label(row) == label and "reward_rates" in row]
        for option in REWARDS:
            series = [{"step": row["step"], "value": row["reward_rates"].get(option)} for row in rows]
            _plot_series(reward_ax, series, "step", "value", option)
        for rate, title in (("shop_reach_rate", "Reached a shop"), ("floor_10_reach_rate", "Reached floor 10"),
                            ("exit_rate", "Shop visits exited"), ("repeat_rate", "Shop visits with repeats"),
                            ("episode_shop_timeout_rate", "Episodes timed out in shop"),
                            ("mean_leave_probability", "Mean Leave probability")):
            series = [{"step": row["step"], "value": row["shop_rates"].get(rate)} for row in rows]
            _plot_series(shop_ax, series, "step", "value", title)
        reward_ax.set(title=f"{label} · rewards", xlabel="Environment steps", ylabel="Selection share", ylim=(-0.025, 1.025))
        shop_ax.set(title=f"{label} · shop decisions", xlabel="Environment steps", ylabel="Rate", ylim=(-0.025, 1.025))
    _finish_figure(plt, fig, destination)


def _display(value, spec=""):
    return "—" if value is None else format(value, spec)


def _markdown_row(values):
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def _build_description(snapshot):
    slots = []
    for index, slot in enumerate(snapshot.get("slots", ()), 1):
        content = "locked" if slot.get("locked") else (
            f"{slot.get('skill') or 'empty'} × {slot.get('uses', 0)} at {slot.get('mana_cost', 0)} mana")
        slots.append(f"{index} {content}")
    scalars = {key.replace("_", " "): value for key, value in snapshot.items()
               if key not in {"slots", "stats", "extra_stats"}}
    scalars.update({key.upper(): value for key, value in snapshot.get("stats", {}).items()})
    scalars.update(snapshot.get("extra_stats", {}))
    return ", ".join(slots + [f"{name} {value}" for name, value in scalars.items()]) or "—"


def _decision_sections(row):
    stats = row["decision_stats"]
    output = [f"## {_label(row)} · stage {row['stage']} · step {row['step']:,}", "",
              "### Combat", "", "| Skill | Available decisions | Selected | Selection when available | Mean model probability when available |",
              "|---|---:|---:|---:|---:|"]
    for skill, data in stats["combat"].items():
        output.append(_markdown_row((skill, data["available"], data["selected"],
                                     _display(data["selection_rate"], ".2%"), _display(data["mean_probability"], ".2%"))))
    reward = stats["reward"]
    output.extend(("", "### Rewards", "", "Each cell shows selected / available, then the share of reward selections. "
                   "Selection when available is the first count divided by the second.", "",
                   _markdown_row(("Floor", *REWARDS)), "|---|---:|---:|---:|---:|---:|"))
    for label, counts in [("All floors", reward["selected"]), *reward["by_floor"].items()]:
        total = sum(counts.values())
        available = reward.get("available", {}) if label == "All floors" else reward.get("available_by_floor", {}).get(label, {})
        cells = [f"{counts[option]} / {_display(available.get(option))} ({counts[option] / total:.1%})" if total else "0 / 0"
                 for option in REWARDS]
        output.append(_markdown_row((label, *cells)))
    shop = stats["shop"]
    output.extend(("", "### Shop", "", "| Strategic function | Available | Selected | Share | Selected when available | First in visit | Repeated in visit |",
                   "|---|---:|---:|---:|---:|---:|---:|"))
    for function, data in shop["functions"].items():
        output.append(_markdown_row((function, _display(data.get("available")), data["selected"], _display(data["rate"], ".2%"),
                                     _display(data.get("selection_rate"), ".2%"), _display(data["first"]),
                                     _display(data["repeat"]))))
    output.extend(("", f"Completed episodes reaching a shop {_display(shop['shop_reach_rate'], '.2%')}. "
                   f"Reached floor 10 {_display(shop['floor_10_reach_rate'], '.2%')}. "
                   f"Recorded visits {shop['visits']}, exits {_display(shop['exit_rate'], '.2%')}, "
                   f"visits with a repeat {_display(shop['repeat_rate'], '.2%')}, "
                   f"and timed-out visits {_display(shop['timeout_rate'], '.2%')}. "
                   f"Episodes ending in a shop timeout {_display(shop['episode_shop_timeout_rate'], '.2%')}. "
                   f"Mean strategic and engine actions per recorded visit "
                   f"{_display(shop['mean_strategic_actions'], '.2f')} and {_display(shop['mean_engine_actions'], '.2f')}. "
                   f"Mean Leave Shop probability when available {_display(shop['mean_leave_probability'], '.2%')}. "
                   f"Cumulative shop penalty {shop['penalty']:.6f}.", ""))
    for function, changes in shop["changes"].items():
        output.extend((f"#### {function}", "", f"Commits {changes['commits']}, state changes {changes['changed']}, "
                       f"repeated commits {changes['repeated_commits']}, repeated commits with a state change "
                       f"{changes['repeated_changed']}, and total gold spent {changes['gold_spent']:,.0f}.", ""))
        if changes["items"]:
            output.extend(("| Transaction | Count |", "|---|---:|",
                           *(_markdown_row((item, count)) for item, count in changes["items"].items()), ""))
        means_before, means_after = changes["before"]["mean"], changes["after"]["mean"]
        if means_before or means_after:
            output.extend(("| Metric | Mean before | Mean after |", "|---|---:|---:|"))
            for metric in sorted(means_before.keys() | means_after.keys()):
                output.append(_markdown_row((metric.replace("_", " "), _display(means_before.get(metric), ".2f"),
                                             _display(means_after.get(metric), ".2f"))))
            output.append("")
        skills_before, skills_after = changes["before"]["skills"], changes["after"]["skills"]
        if skills_before or skills_after:
            output.extend(("| Skill | Equipped before | Equipped after | Uses before | Uses after | Mean mana cost before | Mean mana cost after |",
                           "|---|---:|---:|---:|---:|---:|---:|"))
            for skill in sorted(skills_before.keys() | skills_after.keys()):
                before, after = skills_before.get(skill, {}), skills_after.get(skill, {})
                output.append(_markdown_row((skill, before.get("equipped", 0), after.get("equipped", 0),
                                             _display(before.get("mean_uses"), ".2f"), _display(after.get("mean_uses"), ".2f"),
                                             _display(before.get("mean_mana_cost"), ".2f"),
                                             _display(after.get("mean_mana_cost"), ".2f"))))
            output.append("")
        if changes["examples"]:
            output.extend(("| Example transaction | Before | After |", "|---|---|---|",
                           *(_markdown_row((example.get("item") or function, _build_description(example["before"]),
                                            _build_description(example["after"]))) for example in changes["examples"]), ""))
    output.extend(("### Events", "", "The downstream differences are conditional observations for episodes that selected each option.", "",
                   "| Option | Available | Selected | Share | Selected when available | Episodes | Mean floor afterwards | Max floor afterwards | Reached 40 | Reached 60 | Clear rate | Defeats | Timeouts |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for option, data in stats["event"]["options"].items():
        output.append(_markdown_row((option, _display(data.get("available")), data["selected"], _display(data["rate"], ".2%"),
                                     _display(data.get("selection_rate"), ".2%"), data["episodes"],
                                     _display(data["mean_floor"], ".2f"), _display(data["max_floor"]),
                                     _display(data["survival"].get("40"), ".2%"),
                                     _display(data["survival"].get("60"), ".2%"),
                                     _display(data["clear_rate"], ".2%"), data["outcomes"].get("death", 0),
                                     data["outcomes"].get("timeout", 0))))
    output.append("")
    return output


def generate_report(db_path, output_dir):
    """Read compact SQLite metrics and write English Markdown reports and plots."""
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
        has_decisions = db.execute("SELECT 1 FROM sqlite_master WHERE name = 'evaluations_with_decisions'").fetchone()
        source = "evaluations_with_decisions" if has_decisions else "evaluations"
        evaluations = _sampled_rows(db, source, project=_decision_trend_projection)
        latest = [json.loads(row[0]) for row in db.execute(f"""
            SELECT payload FROM {source} WHERE id IN
            (SELECT MAX(id) FROM evaluations GROUP BY policy, split) ORDER BY policy, split
        """)]
        stages = [json.loads(row[0]) for row in db.execute(f"""
            SELECT payload FROM {source} WHERE id IN
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
    detailed = [row for row in stages if "decision_stats" in row]
    if detailed:
        _plot_decision_trends(plt, evaluations, output / "decision_trends.png")
        report_sections = ["# Detailed evaluation decisions", "",
                           "Tables show the latest panel for each curriculum stage, policy and split. "
                           "Policy probabilities are conditional on the legal action mask. The trends include the sampled evaluation panels.", "",
                           "![Reward and shop trends](decision_trends.png)", ""]
        for row in detailed:
            report_sections.extend(_decision_sections(row))
        (output / "decision_statistics.md").write_text("\n".join(report_sections), encoding="utf-8")
    report = output / "training_report.md"
    last = updates[-1] if updates else {}
    progress_text = (f"Recorded training progress reached {last['step']:,} environment steps."
                     if last else "The database contains no training updates.")
    if last.get("elapsed") is not None:
        progress_text += f" Recorded training wall-clock time was {last['elapsed'] / 3600:.3f} hours."
    if last.get("throughput") is not None:
        progress_text += f" The last measured throughput was {last['throughput']:.1f} steps/s."
    timed = [row for row in stages if row.get('evaluation_seconds') is not None]
    timing = [f"- {_label(row)}, stage {row['stage']}, step {row['step']:,}. "
              f"Training {row.get('training_seconds', row['elapsed']):.2f}s. Evaluation panel {row['evaluation_seconds']:.2f}s. "
              f"Cumulative evaluation {row['evaluation_elapsed']:.2f}s. "
              f"Policy {row['policy_seconds']:.2f}s and workers/IPC {row['worker_seconds']:.2f}s. "
              f"{row['eval_actions']:,} actions across {row['decision_batches']:,} batches."
              for row in timed]
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
        *(["## Evaluation timings", "", *timing, ""] if timing else []),
        "## Learning curves", "", "![Learning curves](learning_curves.png)", "",
        "## Floor-wise survival", "", "![Floor-wise survival](floor_survival.png)", "",
        "## Policy heads and updates", "", "![Policy head statistics](head_statistics.png)", "",
        "## Hardware utilization", "", "![Hardware utilization](hardware_metrics.png)", "",
        *(["## Evaluation decisions", "", "[Detailed decisions and trends](decision_statistics.md)", ""] if detailed else []),
    ]), encoding="utf-8")
    return report
