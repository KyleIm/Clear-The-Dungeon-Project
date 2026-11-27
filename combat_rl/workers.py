"""Independent simulator workers. PyTorch runs only in the parent process."""

import multiprocessing as mp
from time import perf_counter
import traceback

from .environment import Run


def _apply(runs, slot, operation, arg):
    if operation == 'reset':
        run = runs[slot] = Run(*arg)
        return run.observation()
    if operation == 'restore':
        runs[slot] = arg
        return None
    run = runs[slot]
    if operation == 'snapshot':
        return run
    if operation in ('step', 'random'):
        before = run.actions
        reward = run.step(run.random_action() if operation == 'random' else arg)
        return (reward, run.actions - before,
                None if run.done else run.observation(), run.result() if run.done else None)
    raise ValueError(f'Unknown worker operation {operation}')


def serve(pipe, grouped, profiling):
    runs = {}
    with pipe:
        try:
            while True:
                operation, arg = pipe.recv()
                if operation == 'close':
                    return
                timed = profiling and operation in ('step', 'random')
                if timed:
                    started = perf_counter()
                if grouped:
                    result = {slot: _apply(runs, slot, operation, value) for slot, value in arg}
                else:
                    result = _apply(runs, 0, operation, arg)
                if timed:
                    result = result, perf_counter() - started
                pipe.send((True, result))
        except EOFError:
            return
        except Exception:
            pipe.send((False, traceback.format_exc()))


class Workers:
    def __init__(self, count, games_per_worker=1, profile=None):
        if count < 1 or games_per_worker < 1:
            raise ValueError('Worker count and games per worker must be positive')
        self.count = count
        self.games_per_worker = games_per_worker
        self.profile = profile
        self.pairs = []

    def __enter__(self):
        context = mp.get_context('spawn')
        try:
            for _ in range(self.count):
                parent, child = context.Pipe()
                process = context.Process(target=serve,
                                          args=(child, self.games_per_worker > 1, self.profile is not None),
                                          daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.pairs.append((parent, process))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def call(self, operation, items):
        if not items:
            return {}
        timed = self.profile is not None and operation in ('step', 'random')
        if timed:
            started = perf_counter()
        grouped = self.games_per_worker > 1
        if grouped:
            requests = {}
            for slot, arg in items.items():
                if not 0 <= slot < self.count * self.games_per_worker:
                    raise IndexError(f'Invalid simulator slot {slot}')
                requests.setdefault(slot % self.count, []).append((slot, arg))
        else:
            requests = items
        for worker, arg in requests.items():
            self.pairs[worker][0].send((operation, arg))
        results = {}
        compute_times = []
        for worker in requests:
            ok, value = self.pairs[worker][0].recv()
            if not ok:
                raise RuntimeError(f'Simulator worker {worker} failed:\n{value}')
            if timed:
                value, elapsed = value
                compute_times.append(elapsed)
            if grouped:
                results.update(value)
            else:
                results[worker] = value
        if timed:
            # Max compute approximates the critical path in a synchronous round.
            # The residual includes IPC, scheduling and synchronization delays.
            wall = perf_counter() - started
            critical = max(compute_times)
            self.profile.add('worker_round', wall)
            self.profile.add('cpu_simulation', critical)
            self.profile.add('cpu_simulation_total', sum(compute_times))
            self.profile.add('worker_ipc_sync', max(0.0, wall - critical))
        return {slot: results[slot] for slot in items} if grouped else results

    def __exit__(self, *_):
        for pipe, _ in self.pairs:
            try:
                pipe.send(('close', None))
            except OSError:
                pass
            pipe.close()
        for _, process in self.pairs:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
