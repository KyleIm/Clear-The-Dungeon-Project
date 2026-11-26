"""Reproducible local Uniform Random Policy experiment."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import sqlite3
from statistics import mean, median
import subprocess
from time import perf_counter
from typing import NamedTuple

from simulator.data_loader import GameData
from simulator.game import Game

ROOT = Path(__file__).resolve().parents[1]
RANK_FLOORS = (5, 7, 10, 15, 20)


class Episode(NamedTuple):
    game_id: int
    game_seed: int
    policy_seed: int
    initial_area: str | None
    initial_skills: str
    outcome: str
    floor: int | None
    death_monster_id: int | None
    death_monster: str | None
    actions: int
    runtime_s: float
    detail: str | None


def experiment_seeds(games, master_seed):
    """Sample unique signed-SQLite-safe seeds for both independent RNG streams."""
    values = random.Random(master_seed).sample(range(1 << 62), games * 2)
    return [(index + 1, values[2 * index], values[2 * index + 1]) for index in range(games)]


def episode(game, job, action_cap, timeout_s):
    game_id, game_seed, policy_seed = job
    policy = random.Random(policy_seed)
    start = perf_counter()
    actions = 0
    area, skills, floor = None, "[]", None
    outcome, detail = "error", None
    monster_id, monster = None, None
    try:
        state = game.reset(game_seed)
        area, floor = state["area"], state["floor"]
        slots = [name for name in state["player"]["current_skills"] if name != "lock"]
        skills = json.dumps(slots, separators=(",", ":"))
        while not game.is_terminal():
            if actions >= action_cap:
                outcome, detail = "timeout", "action_limit"
                break
            if perf_counter() - start >= timeout_s:
                outcome, detail = "timeout", "wall_limit"
                break
            choice = policy.choice(game.get_legal_actions())
            actions += 1
            state = game.step(choice)
            floor = max(floor, state["floor"])
        else:
            outcome = "cleared" if game.is_cleared() else "game_over"
            if outcome == "game_over":
                monster_id = state["monster"]["monster_id"]
                monster = state["monster"]["name"]
    except Exception as error:
        outcome, monster_id, monster = "error", None, None
        detail = f"{type(error).__name__}: {error}"
    return Episode(game_id, game_seed, policy_seed, area, skills, outcome, floor,
                   monster_id, monster, actions, perf_counter() - start, detail)


def worker(jobs, action_cap, timeout_s):
    game = Game(GameData())
    READY.wait(timeout=60)
    return [episode(game, job, action_cap, timeout_s) for job in jobs]


def initialize_worker(barrier):
    global READY
    READY = barrier


def execute(jobs, workers, action_cap, timeout_s):
    # One nonempty shard per process and a barrier ensure all requested workers run.
    context = mp.get_context("spawn")
    barrier = context.Barrier(workers)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=initialize_worker, initargs=(barrier,),
    ) as pool:
        partitions = [pool.submit(worker, jobs[i::workers], action_cap, timeout_s)
                      for i in range(workers)]
        return sorted(row for task in partitions for row in task.result())


def save_results(db, settings, results):
    with db:
        db.execute("CREATE TABLE run (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
        db.executemany("INSERT INTO run VALUES (?, ?)",
                       ((name, json.dumps(value)) for name, value in settings.items()))
        db.execute("""
            CREATE TABLE episodes (
                game_id INTEGER PRIMARY KEY, game_seed INTEGER UNIQUE NOT NULL,
                policy_seed INTEGER UNIQUE NOT NULL, initial_area TEXT, initial_skills TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK(outcome IN ('game_over','cleared','timeout','error')),
                floor INTEGER CHECK(floor BETWEEN 1 AND 100),
                death_monster_id INTEGER, death_monster TEXT,
                actions INTEGER NOT NULL CHECK(actions >= 0),
                runtime_s REAL NOT NULL CHECK(runtime_s >= 0), detail TEXT,
                CHECK((outcome = 'game_over') = (death_monster_id IS NOT NULL))
            )
        """)
        db.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", results)


def floor_summary(rows):
    return {
        "games": len(rows),
        "eligible": sum(row["floor"] is not None for row in rows),
        "mean_floor": mean(row["floor"] for row in rows if row["floor"] is not None)
        if any(row["floor"] is not None for row in rows) else None,
        **{f"reached_{floor}": sum((row["floor"] or 0) >= floor for row in rows)
           for floor in RANK_FLOORS},
        "clears": sum(row["outcome"] == "cleared" for row in rows),
        "incomplete": sum(row["outcome"] in ("timeout", "error") for row in rows),
    }


def group_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        value = key(row)
        if value is not None:
            groups[value].append(row)
    return [{"group": label, **floor_summary(members)} for label, members in sorted(groups.items())]


def tables(db):
    db.row_factory = sqlite3.Row
    rows = list(db.execute("SELECT * FROM episodes ORDER BY game_id"))
    outcomes = Counter(row["outcome"] for row in rows)
    distribution = Counter(row["floor"] for row in rows if row["floor"] is not None)
    floors = []
    for floor in range(1, 101):
        entrants = sum((row["floor"] or 0) >= floor for row in rows)
        deaths = sum(row["floor"] == floor and row["outcome"] == "game_over" for row in rows)
        incomplete = sum(row["floor"] == floor and row["outcome"] in ("timeout", "error") for row in rows)
        passed = sum((row["floor"] or 0) > floor or (floor == 100 and row["outcome"] == "cleared") for row in rows)
        floors.append({"floor": floor, "entrants": entrants, "advanced": passed,
                       "deaths": deaths, "incomplete": incomplete})
    death = list(db.execute("""
        SELECT death_monster_id AS id, death_monster AS name, COUNT(*) AS deaths
        FROM episodes WHERE outcome = 'game_over'
        GROUP BY death_monster_id, death_monster ORDER BY deaths DESC, id
    """))
    slots = {row["game_id"]: json.loads(row["initial_skills"]) for row in rows}

    def slot_group(row, indexes):
        skills = slots[row["game_id"]]
        return " / ".join(skills[i] for i in indexes) if len(skills) == 4 else None

    return {
        "run": {row["key"]: json.loads(row["value_json"]) for row in db.execute("SELECT * FROM run")},
        "outcomes": {name: outcomes[name] for name in ("game_over", "cleared", "timeout", "error")},
        "floor_distribution": [{"floor": floor, "reached": count} for floor, count in sorted(distribution.items())],
        "floor_survival": floors,
        "death_monsters": [dict(row) for row in death],
        "initial_areas": group_by(rows, lambda row: row["initial_area"]),
        "initial_decks": group_by(rows, lambda row: slot_group(row, range(4))),
        "initial_starters": group_by(rows, lambda row: slot_group(row, (0,))),
        "initial_attack_pairs": group_by(
            rows, lambda row: " / ".join(sorted(slots[row["game_id"]][1:3])) or None,
        ),
        "initial_rank_skills": group_by(rows, lambda row: slot_group(row, (3,))),
        "timing": {key: {"mean": mean(values), "median": median(values),
                         "p95": sorted(values)[(95 * len(values) + 99) // 100 - 1], "max": max(values)}
                   for key, values in ((k, [row[k] for row in rows]) for k in ("actions", "runtime_s"))},
    }


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    limit = max((row["floor"] for row in summary["floor_distribution"]), default=1)
    visible = summary["floor_survival"][:limit + 1]
    distribution = {row["floor"]: row["reached"] for row in summary["floor_distribution"]}
    games = summary["run"]["games"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 6.5), layout="constrained")
    a.bar(range(1, limit + 1), [distribution.get(i, 0) for i in range(1, limit + 1)], color="#315f72")
    a.set(title="Furthest floor reached", xlabel="Floor", ylabel="Episodes")
    b.plot([row["floor"] for row in visible], [100 * row["entrants"] / games for row in visible],
           color="#a04433", marker="o", markersize=3)
    b.set(title="Observed survival curve", xlabel="Floor reached", ylabel="Share of all runs (%)", ylim=(0, 104))
    for ax in (a, b):
        ax.grid(axis="y", alpha=.18)
    fig.suptitle(f"Uniform random policy | {games:,} independent runs", fontweight="bold")
    fig.savefig(output / "random_1000_floors.png")
    plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.1), layout="constrained")
    deaths = list(reversed(summary["death_monsters"][:12]))
    labels = [f'{row["name"]} (#{row["id"]})' if row["name"] == "Ella" else row["name"] for row in deaths]
    a.barh(labels, [row["deaths"] for row in deaths], color="#a04433")
    a.set(title="Top 12 opponents at defeat", xlabel="Game-over episodes")
    areas = sorted(summary["initial_areas"], key=lambda row: row["reached_5"] / row["games"])
    labels = [f'{row["group"]} (n={row["games"]})' for row in areas]
    a5 = [100 * row["reached_5"] / row["games"] for row in areas]
    a10 = [100 * row["reached_10"] / row["games"] for row in areas]
    y = list(range(len(areas)))
    b.barh([i - .18 for i in y], a5, height=.34, label="Reached floor 5", color="#315f72")
    b.barh([i + .18 for i in y], a10, height=.34, label="Reached floor 10", color="#cb8b42")
    b.set(yticks=y, yticklabels=labels, title="Survival by starting area", xlabel="Share of area cohort (%)")
    b.legend(loc="lower right")
    for ax in (a, b):
        ax.grid(axis="x", alpha=.18)
    fig.savefig(output / "random_1000_monsters_areas.png")
    plt.close(fig)

    ranks = sorted(summary["initial_rank_skills"], key=lambda row: row["reached_5"] / row["games"])
    pairs = sorted((row for row in summary["initial_attack_pairs"] if row["games"] >= 20),
                   key=lambda row: row["reached_5"] / row["games"])
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .27 * len(pairs) + 1.8)), layout="constrained")
    for ax, groups, title in zip(axes, (ranks, pairs), ("Starting rank skill", "Starting attack pair (n ≥ 20)")):
        ax.barh([f'{r["group"]} (n={r["games"]})' for r in groups],
                [100 * r["reached_5"] / r["games"] for r in groups], color="#315f72")
        ax.set(title=title, xlabel="Reached floor 5 (%)")
        ax.grid(axis="x", alpha=.18)
    fig.savefig(output / "random_1000_skills.png")
    plt.close(fig)


def revision():
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                              capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--master-seed", type=int, default=314159)
    parser.add_argument("--action-cap", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (1 <= args.workers <= min(args.games, 61) and args.action_cap > 0 and args.timeout_seconds > 0):
        parser.error("positive games, action and time limits and 1–61 workers (no more than games) required")
    stamp = datetime.now(timezone.utc)
    output = args.output or ROOT / "data" / f"baseline_{stamp:%Y%m%dT%H%M%S%fZ}"
    output.mkdir(parents=True, exist_ok=False)

    measured = perf_counter()
    results = execute(experiment_seeds(args.games, args.master_seed), args.workers,
                      args.action_cap, args.timeout_seconds)
    settings = {
        "started_at_utc": stamp.isoformat(), "games": args.games, "workers": args.workers,
        "master_seed": args.master_seed, "action_cap": args.action_cap, "timeout_seconds": args.timeout_seconds,
        "simulation_wall_s": perf_counter() - measured, "python": platform.python_version(),
        "os": platform.system(), "simulator_checkout": revision(),
        "policy": "uniform choice among all current legal actions",
    }
    database = output / "baseline.sqlite3"
    with sqlite3.connect(database) as db:
        save_results(db, settings, results)
        summary = tables(db)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(summary, output)
    print(f"Results saved to {output}")
    print(json.dumps(summary["outcomes"]))


if __name__ == "__main__":
    mp.freeze_support()
    main()
