"""Independent simulator workers. PyTorch runs only in the parent process."""

import multiprocessing as mp
import traceback

from .environment import Run


def serve(pipe):
    run = None
    with pipe:
        try:
            while True:
                operation, arg = pipe.recv()
                if operation == 'close':
                    return
                if operation == 'reset':
                    run = Run(*arg)
                    result = run.observation()
                elif operation == 'restore':
                    run = arg
                    result = None
                elif operation == 'snapshot':
                    result = run
                elif operation in ('step', 'random'):
                    before = run.actions
                    reward = run.step(run.random_action() if operation == 'random' else arg)
                    result = (reward, run.actions - before,
                              None if run.done else run.observation(), run.result() if run.done else None)
                else:
                    raise ValueError(f'Unknown worker operation {operation}')
                pipe.send((True, result))
        except EOFError:
            return
        except Exception:
            pipe.send((False, traceback.format_exc()))


class Workers:
    def __init__(self, count):
        self.count = count
        self.pairs = []

    def __enter__(self):
        context = mp.get_context('spawn')
        try:
            for _ in range(self.count):
                parent, child = context.Pipe()
                process = context.Process(target=serve, args=(child,), daemon=True)
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
        for slot, arg in items.items():
            self.pairs[slot][0].send((operation, arg))
        results = {}
        for slot in items:
            ok, value = self.pairs[slot][0].recv()
            if not ok:
                raise RuntimeError(f'Simulator worker {slot} failed:\n{value}')
            results[slot] = value
        return results

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
