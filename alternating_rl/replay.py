"""Bounded, checkpoint-aligned SQLite replay for shop transitions."""

from operator import index

import numpy as np


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS shop_replay (
    id INTEGER PRIMARY KEY,
    cycle INTEGER NOT NULL CHECK(cycle >= 0),
    version INTEGER NOT NULL CHECK(version >= 0),
    source TEXT NOT NULL,
    head INTEGER NOT NULL CHECK(head BETWEEN 0 AND 2),
    state BLOB NOT NULL,
    state_mask BLOB NOT NULL,
    state_actions INTEGER NOT NULL CHECK(state_actions > 0),
    action INTEGER NOT NULL CHECK(action >= 0 AND action < state_actions),
    next_head INTEGER CHECK(next_head BETWEEN 0 AND 2),
    next_state BLOB,
    next_mask BLOB,
    next_actions INTEGER CHECK(next_actions > 0),
    reward INTEGER NOT NULL CHECK(reward IN (0,1)),
    terminal INTEGER NOT NULL CHECK(terminal IN (0,1)),
    CHECK((terminal = 1 AND next_state IS NULL AND next_head IS NULL) OR
          (terminal = 0 AND next_state IS NOT NULL AND next_head IS NOT NULL AND reward = 0))
)
'''


def _encode(observation):
    features, head, mask = observation
    features = np.asarray(features, dtype='<f4')
    mask = np.asarray(mask, dtype=np.bool_)
    if features.ndim != 1 or mask.ndim != 1 or not np.isfinite(features).all() or not mask.any():
        raise ValueError('A replay observation needs finite features and legal actions')
    return index(head), features.tobytes(), np.packbits(mask).tobytes(), len(mask)


def _decode(head, features, mask, size):
    return np.frombuffer(features, dtype='<f4'), head, np.unpackbits(np.frombuffer(mask, dtype=np.uint8),
                                                                      count=size).astype(np.bool_)


def _decode_row(row):
    (_id, cycle, version, source, head, state, mask, size, action,
     next_head, next_state, next_mask, next_size, reward, terminal) = row
    return dict(state=_decode(head, state, mask, size), action=action,
                next_state=None if terminal else _decode(next_head, next_state, next_mask, next_size),
                reward=reward, terminal=bool(terminal), version=version, cycle=cycle, source=source)


class Replay:
    """Keep newest transitions, restricting draws to the configured recent cycles.

    Commit ownership remains with the enclosing experiment store. Prune only
    after saving a checkpoint at the supplied mark. This preserves the replay
    window needed by the latest checkpoint if an interrupted run writes extra
    rows before a new checkpoint is safely on disk.
    """

    def __init__(self, db, capacity, previous_cycles, *, recent_fraction=0.75, recent_window=0.25):
        if capacity < 1 or previous_cycles < 0:
            raise ValueError('Replay needs a positive capacity and nonnegative cycle lookback')
        if not 0 <= recent_fraction <= 1 or not 0 < recent_window <= 1:
            raise ValueError('Replay recency fractions are outside their supported range')
        self.db = db
        self.capacity = capacity
        self.previous_cycles = previous_cycles
        self.recent_fraction = recent_fraction
        self.recent_window = recent_window
        db.execute(_SCHEMA)
        db.execute('CREATE INDEX IF NOT EXISTS shop_replay_cycle_id ON shop_replay(cycle, id)')

    def append(self, row):
        terminal = bool(row['terminal'])
        if terminal != (row['next_state'] is None):
            raise ValueError('Only terminal transitions may omit the next observation')
        action = index(row['action'])
        mask = row['state'][2]
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('A replay action must have been legal at selection time')
        reward = row['reward']
        if reward not in (0, 1) or (reward and not terminal):
            raise ValueError('Only a resolved terminal milestone may earn a reward')
        initial = _encode(row['state'])
        following = (None,) * 4 if terminal else _encode(row['next_state'])
        self.db.execute('''
            INSERT INTO shop_replay
                (cycle,version,source,head,state,state_mask,state_actions,action,
                 next_head,next_state,next_mask,next_actions,reward,terminal)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (index(row['cycle']), index(row['version']), row.get('source', 'dqn'),
              *initial, action, *following, int(reward), int(terminal)))

    def mark(self):
        return self.db.execute('SELECT COALESCE(MAX(id),0) FROM shop_replay').fetchone()[0]

    def restore(self, mark):
        self.db.execute('DELETE FROM shop_replay WHERE id > ?', (mark,))

    def prune(self, mark):
        self.db.execute('DELETE FROM shop_replay WHERE id < ?', (max(1, mark - self.capacity + 1),))

    def sample(self, size, rng, current_cycle, current_version):
        if size < 1:
            raise ValueError('A replay minibatch size must be positive')
        oldest_cycle = max(0, current_cycle - self.previous_cycles)
        ceiling = self.mark()
        earliest = self.db.execute('''
            SELECT id FROM shop_replay WHERE cycle BETWEEN ? AND ? ORDER BY cycle, id LIMIT 1
        ''', (oldest_cycle, current_cycle)).fetchone()
        if earliest is None:
            return []
        floor = max(earliest[0], ceiling - self.capacity + 1)
        if floor > ceiling:
            return []
        span = ceiling - floor + 1
        recent_floor = max(floor, ceiling - max(1, int(span * self.recent_window)) + 1)
        # IDs remain contiguous apart from trimming the prefix or restoring the suffix.
        # Sample keys directly to avoid ORDER BY RANDOM over multi-megabyte state blobs.
        wanted = min(size, span)
        keys = {}
        for _ in range(wanted * 4):
            if len(keys) >= wanted:
                break
            start = recent_floor if rng.random() < self.recent_fraction else floor
            keys[start + int(rng.random() * (ceiling - start + 1))] = None
        selected = {}
        ids = list(keys)
        for offset in range(0, len(ids), 256):
            chunk = ids[offset:offset + 256]
            placeholders = ','.join('?' for _ in chunk)
            rows = self.db.execute(f'''
                SELECT * FROM shop_replay WHERE id IN ({placeholders})
                AND cycle BETWEEN ? AND ? AND version <= ?
            ''', (*chunk, oldest_cycle, current_cycle, current_version))
            selected.update((row[0], row) for row in rows)
        # Close to capacity or a version boundary, fill from the newest eligible rows.
        if len(selected) < wanted:
            rows = self.db.execute('''
                SELECT * FROM shop_replay WHERE id BETWEEN ? AND ?
                AND cycle BETWEEN ? AND ? AND version <= ? ORDER BY id DESC LIMIT ?
            ''', (floor, ceiling, oldest_cycle, current_cycle, current_version, wanted))
            for row in rows:
                selected.setdefault(row[0], row)
                if len(selected) >= wanted:
                    break
        return [_decode_row(row) for row in selected.values()]

    def summary(self, current_cycle=None):
        ceiling = self.mark()
        groups = self.db.execute('''
            SELECT cycle,head,source,COUNT(*),SUM(terminal),SUM(reward)
            FROM shop_replay WHERE id >= ? AND cycle >= ? GROUP BY cycle,head,source
        ''', (max(1, ceiling - self.capacity + 1),
              0 if current_cycle is None else max(0, current_cycle - self.previous_cycles))).fetchall()
        return dict(count=sum(row[3] for row in groups), groups=[
            dict(cycle=cycle, head=head, source=source, transitions=count,
                 terminals=terminals, successes=successes)
            for cycle, head, source, count, terminals, successes in groups])
