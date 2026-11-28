"""Checkpoint-aligned episode and paired-configuration telemetry."""

import sqlite3

import numpy as np

from combat_rl.report import Store as PPOStore, _encode


class Store(PPOStore):
    def __enter__(self):
        super().__enter__()
        try:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, version INTEGER NOT NULL,
                    head INTEGER NOT NULL, label INTEGER, features BLOB, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS comparison_sampling ON comparisons(version, head, label, id)
                WHERE features IS NOT NULL;
                CREATE TABLE IF NOT EXISTS run_metadata (
                    id INTEGER PRIMARY KEY CHECK (id=1), run_id TEXT NOT NULL
                );
            """)
        except sqlite3.Error:
            self.db.close()
            raise
        return self

    def identity(self, run_id, resume):
        if not resume:
            self._write('INSERT INTO run_metadata VALUES (1,?)', (run_id,))
        elif self.db.execute('SELECT run_id FROM run_metadata').fetchone() != (run_id,):
            raise ValueError('Checkpoint and database belong to different experiments')

    def comparison(self, step, version, head, features, row):
        label = row['label']
        blob = None if label is None or row['noop'] else features.astype('<f2').tobytes()
        self._write('INSERT INTO comparisons(step,version,head,label,features,payload) VALUES (?,?,?,?,?,?)',
                    (step, version, head, label, blob, _encode(row)))

    def sample(self, version, head, batch, window, rng):
        candidates = []
        for label in (-1, 0, 1):
            ids = [row[0] for row in self.db.execute(
                'SELECT id FROM comparisons WHERE version=? AND head=? AND label=? AND features IS NOT NULL '
                'ORDER BY id DESC LIMIT ?', (version, head, label, window))]
            if ids:
                candidates.append(ids)
        # Wait for contrasting outcomes. Oversample minority classes only up to the batch quota.
        if len(candidates) < 2:
            return []
        selected = []
        quota = batch // len(candidates)
        for ids in candidates:
            selected.extend(rng.choices(ids, k=quota))
        data = {}
        for row, features, label in self.db.execute(
                f'SELECT id,features,label FROM comparisons WHERE id IN ({",".join("?" * len(selected))})', selected):
            data[row] = np.frombuffer(features, dtype='<f2').astype(np.float32), label
        rng.shuffle(selected)
        return [data[row] for row in selected]

    def marks(self):
        marks = super().marks()
        marks['comparisons'] = self.db.execute('SELECT coalesce(max(id),0) FROM comparisons').fetchone()[0]
        return marks

    def restore(self, marks):
        if any(self.db.execute(f'SELECT coalesce(max(id),0) FROM {table}').fetchone()[0] < row
               for table, row in marks.items()):
            raise ValueError('Database is missing committed checkpoint telemetry')
        self.db.execute('DELETE FROM comparisons WHERE id > ?', (marks['comparisons'],))
        super().restore(marks)
