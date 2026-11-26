"""Windows-spawn environment workers with explicit reset and checkpoint boundaries."""

from contextlib import contextmanager
import multiprocessing as mp
import traceback

from .environment import DungeonEnv


def _transition(env, command, arg):
    if command == 'legal':
        return env.legal_ids_in_engine_order()
    if command == 'reset':
        return env.reset(seed=arg)
    if command == 'step':
        return env.step(arg)
    raise ValueError(f'Unknown worker command {command!r}')


def _worker(pipe, options):
    env = DungeonEnv(**options)
    with pipe:
        try:
            while True:
                command, arg = pipe.recv()
                if command == 'close':
                    return
                if command == 'snapshot':
                    result = env
                elif command == 'restore':
                    env = arg
                    result = None
                else:
                    result = _transition(env, command, arg)
                pipe.send((True, result))
        except EOFError:
            return
        except Exception:
            pipe.send((False, traceback.format_exc()))


class Vector:
    def __init__(self, workers, options):
        self.envs = [DungeonEnv(**options)] if workers == 1 else []
        self.pipes = []
        self.processes = []
        ctx = mp.get_context('spawn')
        try:
            for _ in range(workers if not self.envs else 0):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, options), daemon=True)
                try:
                    process.start()
                finally:
                    child.close()
                self.pipes.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def call(self, command, args):
        if self.envs:
            result = {}
            for i, arg in args.items():
                if command == 'snapshot':
                    result[i] = self.envs[i]
                elif command == 'restore':
                    self.envs[i] = arg
                    result[i] = None
                else:
                    result[i] = _transition(self.envs[i], command, arg)
            return result
        for i, arg in args.items():
            self.pipes[i].send((command, arg))
        result = {}
        for i in args:
            ok, value = self.pipes[i].recv()
            if not ok:
                raise RuntimeError(f'Environment worker {i} failed\n{value}')
            result[i] = value
        return result

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
            pipe.close()
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()


@contextmanager
def open_vector(workers, options):
    vector = Vector(workers, options)
    try:
        yield vector
    finally:
        vector.close()
