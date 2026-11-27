"""Opt-in timing and low-frequency system resource sampling for local benchmarks."""

from collections import Counter
from contextlib import contextmanager
import csv
import math
import os
import shutil
from statistics import mean
import subprocess
from threading import Event, Thread
from time import perf_counter


class Profile:
    """Accumulate inclusive phase times. Detailed mode enables device synchronization."""

    def __init__(self, detailed=False):
        self.detailed = detailed
        self.totals = Counter()
        self.counts = Counter()

    def add(self, name, seconds):
        self.totals[name] += seconds
        self.counts[name] += 1

    @contextmanager
    def measure(self, name):
        started = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - started)

    def summary(self):
        return {name: {'seconds': self.totals[name], 'calls': count}
                for name, count in sorted(self.counts.items())}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class Resources:
    """Poll CPU and memory for this process tree and GPU metrics for one NVIDIA device."""

    def __init__(self, interval=1.0, gpu_index=0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Resource sampling interval must be positive and finite')
        try:
            import psutil
        except ImportError:
            psutil = None
        self._psutil = psutil
        self._interval = interval
        self._gpu_index = gpu_index
        self._gpu_command = shutil.which('nvidia-smi')
        self._stop = Event()
        self._ready = Event()
        self._thread = Thread(target=self._collect, name='benchmark-resource-sampler', daemon=True)
        self._samples = []

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=4)

    def _gpu(self):
        if self._gpu_command is None:
            return None, None, None
        try:
            result = subprocess.run(
                [self._gpu_command, '-i', str(self._gpu_index),
                 '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=2,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            name, utilization, memory = next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
            return None, None, None
        return name, _number(utilization), _number(memory)

    def _cpu_snapshot(self, root):
        psutil = self._psutil
        rss = 0
        times = {}
        for process in [root, *root.children(recursive=True)]:
            try:
                with process.oneshot():
                    if process.name().lower().removesuffix('.exe') == 'nvidia-smi':
                        continue
                    cpu = process.cpu_times()
                    times[process.pid, process.create_time()] = cpu.user + cpu.system
                    rss += process.memory_info().rss
            except psutil.Error:
                continue
        return times, rss / 2**20

    def _collect(self):
        psutil = self._psutil
        root = psutil.Process() if psutil is not None else None
        previous_times = {}
        previous_time = None
        if root is not None:
            psutil.cpu_percent()
        while not self._stop.is_set():
            cpu = rss = system_cpu = ram = None
            tick = perf_counter()
            if root is not None:
                times, rss = self._cpu_snapshot(root)
                if previous_time is not None:
                    delta = sum(max(0, current - previous_times[key]) for key, current in times.items()
                                if key in previous_times)
                    cpu = 100 * delta / (tick - previous_time) / (os.cpu_count() or 1)
                    system_cpu = psutil.cpu_percent()
                previous_times, previous_time = times, tick
                memory = psutil.virtual_memory()
                ram = (memory.total - memory.available) / 2**20
            gpu_name, gpu, vram = self._gpu()
            self._samples.append((cpu, rss, system_cpu, ram, gpu_name, gpu, vram))
            self._ready.set()
            self._stop.wait(max(0, self._interval - (perf_counter() - tick)))

    def summary(self):
        samples = tuple(self._samples)

        def reduce(index, function):
            values = [row[index] for row in samples if row[index] is not None]
            return function(values) if values else None

        return dict(samples=len(samples), process_tree_cpu_percent_mean=reduce(0, mean),
                    process_tree_rss_mib_peak=reduce(1, max), system_cpu_percent_mean=reduce(2, mean),
                    system_ram_used_mib_peak=reduce(3, max), gpu_name=reduce(4, lambda names: names[0]),
                    gpu_util_percent_mean=reduce(5, mean), gpu_memory_used_mib_peak=reduce(6, max))
