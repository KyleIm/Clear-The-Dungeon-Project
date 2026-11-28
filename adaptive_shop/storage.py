"""Durable, resume-aligned Reward decisions and boundary records."""

from collections import Counter
import json

import numpy as np

from baseline_shop.storage import Store as V1
from combat_rl.report import _encode


class Store(V1):
    def __enter__(self):
        super().__enter__()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS reward_experience (
                id INTEGER PRIMARY KEY, segment INTEGER NOT NULL, shop_version INTEGER NOT NULL,
                game_seed TEXT NOT NULL, source TEXT NOT NULL, action INTEGER NOT NULL,
                heldout INTEGER NOT NULL, censored INTEGER NOT NULL, observation BLOB NOT NULL,
                legal BLOB NOT NULL, progress INTEGER NOT NULL, payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reward_sampling ON reward_experience
            (segment,shop_version,heldout,censored,source,action,id);
            CREATE TABLE IF NOT EXISTS boundaries (
                id INTEGER PRIMARY KEY, step INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL
            );
        """)
        return self

    def episode(self, split, step, row):
        row = dict(row)
        experience = row.pop('_experience', [])
        for i, (action, observation, legal, data) in enumerate(experience):
            final_floor = row['floor']
            meta = dict(data, final_floor=final_floor, cleared=row['outcome']=='cleared',
                        outcome=row['outcome'], policy_seed=row['policy_seed'], reward_ordinal=i)
            self._write("""INSERT INTO reward_experience(segment,shop_version,game_seed,source,
                        action,heldout,censored,observation,legal,progress,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (row['segment'], data['shop_version'], str(row['game_seed']), data['source'],
                         action, int(row['game_seed']%5==0), int(row['outcome']=='timeout'),
                         observation.astype('<f4').tobytes(), legal.astype(np.uint8).tobytes(),
                         final_floor-data['floor'], _encode(meta)))
        super().episode(split, step, row)

    def reward_batch(self, segment, version, heldout, limit, rng):
        rows = []
        quota = max(1,limit//10)
        for source in ('random','ppo'):
            for action in range(5):
                keys = [r[0] for r in self.db.execute("""SELECT id FROM reward_experience
                    WHERE segment=? AND shop_version=? AND heldout=? AND censored=0 AND source=? AND action=?
                    ORDER BY id DESC LIMIT ?""",(segment,version,heldout,source,action,quota*2))]
                for offset in range(0,min(len(keys),quota),256):
                    # Select distinct rows using the checkpointed generator.
                    if offset==0:
                        keys = rng.sample(keys,min(len(keys),quota))
                    chunk = keys[offset:offset+256]
                    rows.extend((np.frombuffer(x,dtype='<f4'),np.frombuffer(mask,dtype=np.bool_),
                                 action,progress,source,json.loads(payload))
                                for x,mask,progress,payload in self.db.execute(
                        f'SELECT observation,legal,progress,payload FROM reward_experience '
                        f'WHERE id IN ({",".join("?"*len(chunk))})',chunk))
        return rows

    def boundary(self, step, data):
        self._write('INSERT INTO boundaries(step,payload) VALUES (?,?)', (step,_encode(data)))

    def marks(self):
        return super().marks() | {t:self.db.execute(f'SELECT coalesce(max(id),0) FROM {t}').fetchone()[0]
                                  for t in ('reward_experience','boundaries')}

    def restore(self, marks):
        # The inherited guard validates every table before deleting the tail.
        super().restore(marks)
        for t in ('reward_experience','boundaries'):
            self.db.execute(f'DELETE FROM {t} WHERE id>?',(marks[t],))
        self.flush()
