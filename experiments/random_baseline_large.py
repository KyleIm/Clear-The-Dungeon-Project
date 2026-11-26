"""Run a resumable uniform-random experiment with compact SQLite telemetry."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing, contextmanager
from hashlib import sha256
from itertools import islice
import json
from math import isfinite
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from unittest.mock import patch

from simulator import combat
from simulator.data_loader import GameData
from simulator.game import Game

from .random_baseline import ROOT, episode

DEFAULT_OUTPUT = ROOT / "data" / "baseline_random_1m"
MASK = (1 << 63) - 1
GENERATOR = "mix63-v1"


def seed_at(master, index):
    # Each transformation is bijective modulo 2**63. The two streams cannot overlap.
    value = (master + 0x9E3779B97F4A7C15 * index) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def batch_jobs(games, batch, size, master):
    return [(i + 1, seed_at(master, 2 * i), seed_at(master, 2 * i + 1))
            for i in range(batch * size, min(games, (batch + 1) * size))]


class Trace:
    def __init__(self):
        self.acquired = {}
        self.equipped = Counter()
        self.grants = Counter()
        self.peak = Counter()
        self.casts = Counter()
        self.uses = {}
        self.encounters = defaultdict(lambda: [0, 0, 0])  # entered, wins, losses

    def start(self, state):
        self.learn(state, state["floor"])
        self.observe_slots(state)
        self.encounters[state["monster"]["monster_id"]][0] += 1

    def learn(self, state, floor):
        for name in state["player"]["learned_skills"]:
            if name != "lock":
                self.acquired.setdefault(name, floor)

    def observe_slots(self, state):
        p = state["player"]
        new = {name: p["memorized_skills"].get(name, 0)
               for name in p["current_skills"] if name != "lock"}
        for name, remaining in new.items():
            if name not in self.uses:
                self.equipped[name] += 1
            self.grants[name] += max(0, remaining - self.uses.get(name, 0))
            self.peak[name] = max(self.peak[name], remaining)
        self.uses = new

    def step(self, before, after):
        if before["phase"] == "reward":
            self.learn(after, before["floor"])
        self.observe_slots(after)
        if before["phase"] == "combat" and after["phase"] != "combat":
            if after["phase"] == "game_over":
                self.encounters[before["monster"]["monster_id"]][2] += 1
            elif after["phase"] in {"reward", "event", "cleared"}:
                self.encounters[before["monster"]["monster_id"]][1] += 1
        if before["phase"] != "combat" and after["phase"] == "combat":
            self.encounters[after["monster"]["monster_id"]][0] += 1

    def skill_rows(self, game_id, skill_ids):
        for name in self.acquired.keys() | self.equipped.keys() | self.casts.keys():
            yield (game_id, skill_ids[name], self.acquired.get(name), self.equipped[name],
                   int(name in self.uses), self.grants[name], self.peak[name],
                   self.uses.get(name, 0), self.casts[name])

    def monster_rows(self, game_id):
        for monster, (encounters, wins, losses) in self.encounters.items():
            yield (game_id, monster, encounters, wins, losses, encounters - wins - losses)


def init_worker():
    global WORKER_GAME, SKILL_IDS
    data = GameData()
    WORKER_GAME = Game(data)
    SKILL_IDS = {name: record["id"] for name, record in data.skills.items()}


def play_batch(batch_id, jobs, action_cap, timeout_s):
    engine = WORKER_GAME
    original = combat.resolve_skill
    results, skills, monsters = [], [], []
    trace = None

    def track(game, actor, foe, name):
        if actor is engine.player:
            trace.casts[name] += 1
        return original(game, actor, foe, name)

    # The probe forwards the original arguments and result without touching engine state or RNG.
    with patch.object(combat, "resolve_skill", track):
        for job in jobs:
            trace = Trace()
            row = episode(engine, job, action_cap, timeout_s, observer=trace)
            results.append(row)
            skills.extend(trace.skill_rows(row.game_id, SKILL_IDS))
            monsters.extend(trace.monster_rows(row.game_id))
    return batch_id, results, skills, monsters


SCHEMA = """
CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, games INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS monsters (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
    game_id INTEGER PRIMARY KEY, game_seed INTEGER NOT NULL, policy_seed INTEGER NOT NULL,
    initial_area TEXT, initial_skills TEXT NOT NULL, outcome TEXT NOT NULL,
    floor INTEGER, death_monster_id INTEGER, death_monster TEXT,
    actions INTEGER NOT NULL, runtime_s REAL NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS episode_skills (
    game_id INTEGER NOT NULL REFERENCES episodes(game_id), skill_id INTEGER NOT NULL REFERENCES skills(id),
    acquired_floor INTEGER, equip_events INTEGER NOT NULL, equipped_at_end INTEGER NOT NULL,
    uses_added INTEGER NOT NULL, peak_uses INTEGER NOT NULL, remaining_uses INTEGER NOT NULL,
    actual_casts INTEGER NOT NULL, PRIMARY KEY (game_id, skill_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS episode_monsters (
    game_id INTEGER NOT NULL REFERENCES episodes(game_id), monster_id INTEGER NOT NULL REFERENCES monsters(id),
    encounters INTEGER NOT NULL, wins INTEGER NOT NULL, losses INTEGER NOT NULL, unresolved INTEGER NOT NULL,
    PRIMARY KEY (game_id, monster_id), CHECK(encounters = wins + losses + unresolved)
) WITHOUT ROWID;
"""


def connect(directory):
    db = sqlite3.connect(directory / "baseline.sqlite3")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    return db


@contextmanager
def run_lock(directory):
    with (directory / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Another runner already owns this experiment directory") from error
        yield


def fingerprints():
    paths = sorted((ROOT / "simulator").glob("*.py")) + sorted((ROOT / "data").glob("*.json"))
    paths += [Path(__file__), ROOT / "experiments" / "random_baseline.py"]
    return {str(path.relative_to(ROOT).as_posix()): sha256(
        path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest() for path in paths}


def initialize(db, settings):
    db.executescript(SCHEMA)
    stored = db.execute("SELECT config FROM run WHERE id=1").fetchone()
    settings = {**settings, "seed_generator": GENERATOR, "fingerprints": fingerprints()}
    # Only runtime parallelism can change without changing the sample or its boundaries.
    settings.pop("workers")
    if stored:
        if json.loads(stored[0]) != settings:
            raise ValueError("Existing experiment settings or code differ. Use the original settings or a new --output")
        return
    data = GameData()
    with db:
        db.execute("INSERT INTO run VALUES (1,?)", (json.dumps(settings, sort_keys=True),))
        db.executemany("INSERT INTO skills VALUES (?,?)",
                       ((record["id"], name) for name, record in data.skills.items()))
        db.executemany("INSERT INTO monsters VALUES (?,?)",
                       ((mid, record["name"]) for mid, record in data.monsters.items()))


def commit_batch(db, result):
    batch_id, episodes, skills, monsters = result
    with db:
        db.execute("INSERT INTO batches VALUES (?,?)", (batch_id, len(episodes)))
        db.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", episodes)
        db.executemany("INSERT INTO episode_skills VALUES (?,?,?,?,?,?,?,?,?)", skills)
        db.executemany("INSERT INTO episode_monsters VALUES (?,?,?,?,?,?)", monsters)


def duration(seconds):
    if seconds is None:
        return "--"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run(directory, settings, max_batches=None):
    directory.mkdir(parents=True, exist_ok=True)
    games, size, workers = (settings[key] for key in ("games", "batch_size", "workers"))
    with run_lock(directory), closing(connect(directory)) as db:
        initialize(db, settings)
        existing = dict(db.execute("SELECT id, games FROM batches"))
        before = done = sum(existing.values())
        print(f"Checkpoint: {done:,}/{games:,} games in {directory / 'baseline.sqlite3'}", flush=True)
        if done == games:
            return True
        missing = (bid for bid in range((games + size - 1) // size) if bid not in existing)
        if max_batches is not None:
            missing = islice(missing, max_batches)
        started = perf_counter()
        printed = -2
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                   initializer=init_worker)
        pending = set()

        def submit_next():
            batch = next(missing, None)
            if batch is not None:
                pending.add(pool.submit(play_batch, batch, batch_jobs(
                    games, batch, size, settings["master_seed"]),
                    settings["action_cap"], settings["timeout_seconds"]))

        try:
            for _ in range(2 * workers):
                submit_next()
            while pending:
                ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    result = future.result()
                    commit_batch(db, result)
                    done += len(result[1])
                    submit_next()
                elapsed = perf_counter() - started
                if elapsed - printed >= 1 or not pending:
                    rate = (done - before) / elapsed
                    eta = (games - done) / rate if rate else None
                    print(f"{done:,}/{games:,} ({done/games:.1%}) | session {duration(elapsed)}"
                          f" | {rate:,.0f} games/s | ETA {duration(eta)}", flush=True)
                    printed = elapsed
        except KeyboardInterrupt:
            print(f"\nPaused. SQLite retains {done:,} committed games. Launch again to resume.", flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return db.execute("SELECT coalesce(sum(games),0) FROM batches").fetchone()[0] == games


def summarize(db):
    db.row_factory = sqlite3.Row
    outcomes = {r[0]: r[1] for r in db.execute("SELECT outcome,count(*) FROM episodes GROUP BY outcome")}
    buckets = dict(db.execute("SELECT floor,count(*) FROM episodes WHERE floor IS NOT NULL GROUP BY floor"))
    defeats = dict(db.execute("SELECT floor,count(*) FROM episodes WHERE outcome='game_over' GROUP BY floor"))
    incomplete = dict(db.execute("SELECT floor,count(*) FROM episodes WHERE outcome IN ('timeout','error') GROUP BY floor"))
    floors = []
    entrants = sum(buckets.values())
    for floor in range(1, 101):
        advanced = entrants - buckets.get(floor, 0)
        floors.append({"floor": floor, "entrants": entrants, "advanced": advanced if floor < 100
                       else outcomes.get("cleared", 0), "deaths": defeats.get(floor, 0),
                       "incomplete": incomplete.get(floor, 0)})
        entrants = advanced
    monster_rows = [dict(row) for row in db.execute("""
        SELECT m.id, m.name, sum(e.encounters) encounters, sum(e.wins) wins,
               sum(e.losses) losses, sum(e.unresolved) unresolved
        FROM episode_monsters e JOIN monsters m ON m.id=e.monster_id GROUP BY e.monster_id
        ORDER BY losses DESC, m.id
    """)]
    skill_rows = [dict(row) for row in db.execute("""
        SELECT s.id, s.name, sum(t.acquired_floor IS NOT NULL) acquired, sum(t.equip_events>0) equipped,
               sum(t.equipped_at_end) equipped_at_end, sum(t.uses_added) uses_added,
               max(t.peak_uses) peak_uses, sum(t.remaining_uses) remaining_uses,
               sum(t.actual_casts) casts, sum(t.actual_casts>0) cast_games,
               sum(t.acquired_floor IS NOT NULL AND coalesce(e.floor,0)>=5) acquired_floor5,
               sum(t.acquired_floor IS NOT NULL AND coalesce(e.floor,0)>=10) acquired_floor10,
               sum(t.equip_events>0 AND coalesce(e.floor,0)>=5) equipped_floor5,
               sum(t.equip_events>0 AND coalesce(e.floor,0)>=10) equipped_floor10,
               sum(t.actual_casts>0 AND coalesce(e.floor,0)>=5) cast_floor5,
               sum(t.actual_casts>0 AND coalesce(e.floor,0)>=10) cast_floor10,
               sum(t.acquired_floor IS NOT NULL AND e.outcome='cleared') acquired_clears,
               sum(t.equip_events>0 AND e.outcome='cleared') equipped_clears,
               sum(t.actual_casts>0 AND e.outcome='cleared') cast_clears
        FROM episode_skills t JOIN skills s ON s.id=t.skill_id JOIN episodes e USING (game_id)
        GROUP BY t.skill_id ORDER BY casts DESC, s.id
    """)]
    acquired_by_floor = [dict(row) for row in db.execute("""
        SELECT skill_id, acquired_floor, count(*) AS games FROM episode_skills
        WHERE acquired_floor IS NOT NULL GROUP BY skill_id, acquired_floor ORDER BY skill_id, acquired_floor
    """)]
    timing = dict(db.execute("SELECT avg(actions) mean_actions, avg(runtime_s) mean_runtime_s FROM episodes").fetchone())
    return {"outcomes": outcomes, "floors": floors, "skills": skill_rows, "monsters": monster_rows,
            "acquisition_by_floor": acquired_by_floor, "timing": timing}


def report(directory):
    with closing(connect(directory)) as db:
        data = summarize(db)
    (directory / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    games = sum(data["outcomes"].values())
    lines = [
        "# Uniform Random Policy Results", "",
        f"Completed episodes: {games:,}.", "",
        "| Outcome | Episodes |", "|---|---:|",
    ]
    lines += [f"| {name} | {data['outcomes'].get(name, 0):,} |"
              for name in ("cleared", "game_over", "timeout", "error")]
    timing = data["timing"]
    lines += ["", f"Mean actions: {timing['mean_actions']:.2f}. "
              f"Mean episode runtime: {timing['mean_runtime_s']:.4f} seconds.",
              "", "## Floor survival", "", "| Floor | Reached | Of all runs | Advanced | Defeats | Incomplete |",
              "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['floor']} | {r['entrants']:,} | {r['entrants']/games:.2%} | {r['advanced']:,} | "
              f"{r['deaths']:,} | {r['incomplete']:,} |" for r in data["floors"]]
    lines += ["", "![Floor survival](floor_survival.png)", "", "## Monster encounters", "",
              "| ID | Monster | Encounters | Wins | Losses | Unresolved |", "|---:|---|---:|---:|---:|---:|"]
    lines += [f"| {r['id']} | {r['name']} | {r['encounters']:,} | {r['wins']:,} | {r['losses']:,} | "
              f"{r['unresolved']:,} |" for r in data["monsters"]]
    lines += ["", "## Skill cohorts", "",
              "Percentages below use the number of episodes in each cohort. Later acquisition requires "
              "surviving long enough to obtain the skill, so these rates describe association.",
              "", "| Skill | Acquired | Equipped | Cast episodes | Casts | Peak uses | F5 / F10 if acquired | "
              "F5 / F10 if equipped | F5 / F10 if cast |", "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for r in data["skills"]:
        rates = []
        for group, count in (("acquired", r["acquired"]), ("equipped", r["equipped"]), ("cast", r["cast_games"])):
            rates.append(f"{r[group+'_floor5']/count:.1%} / {r[group+'_floor10']/count:.1%}" if count else "—")
        lines.append(f"| {r['name']} | {r['acquired']:,} | {r['equipped']:,} | {r['cast_games']:,} | "
                     f"{r['casts']:,} | {r['peak_uses']} | " + " | ".join(rates) + " |")
    lines += ["", "Acquisition-floor counts, clear counts by cohort, and usage totals are in "
              "summary.json. Episode-level summaries are in baseline.sqlite3.",
              "", "![Skill cohorts](skill_cohorts.png)"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(data, directory, games)


def plot(data, directory, games):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = [r for r in data["floors"] if r["entrants"]]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.step([r["floor"] for r in floors], [100*r["entrants"]/games for r in floors], where="post")
    ax.set(xlabel="Floor reached", ylabel="Share of episodes (%)", title=f"Floor survival · {games:,} episodes")
    ax.grid(alpha=.2)
    fig.savefig(directory / "floor_survival.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    cohorts = list(reversed([r for r in data["skills"] if r["cast_games"]][:15]))
    ax.barh([f"{r['name']} (n={r['cast_games']:,})" for r in cohorts],
            [100*r["cast_floor5"]/r["cast_games"] for r in cohorts], color="#315f72")
    ax.set(xlabel="Reached floor 5 among episodes casting the skill (%)", title="Most frequently cast skills")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(directory / "skill_cohorts.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--master-seed", type=int, default=314159)
    parser.add_argument("--action-cap", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-batches", type=int, help="Stop after committing this many new batches")
    options = vars(parser.parse_args())
    output, max_batches = options.pop("output"), options.pop("max_batches")
    if not (0 < options["games"] <= (1 << 62) and 1 <= options["workers"] <= 61
            and 1 <= options["batch_size"] <= 4096 and options["action_cap"] > 0
            and isfinite(options["timeout_seconds"]) and options["timeout_seconds"] > 0
            and (max_batches is None or max_batches > 0)):
        parser.error("Invalid episode, worker, batch or timeout settings")
    if run(output, options, max_batches):
        report(output)
        print(f"Complete. Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
