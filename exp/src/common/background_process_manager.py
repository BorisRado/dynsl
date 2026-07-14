import time
import threading
import multiprocessing
import multiprocessing as mp
from datetime import datetime


def cpu_intensive_task(run_event, load=0.7, period=0.05):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import numpy as np
    os.sched_setaffinity(0, {3})                 # own ONE core
    a = np.random.randn(64, 64).astype(np.float32)   # allocate ONCE
    b = np.random.randn(64, 64).astype(np.float32)
    c = np.empty((64, 64), dtype=np.float32)          # reuse output buffer
    busy = load * period
    while True:
        run_event.wait()                # blocks while paused
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < busy:
            np.matmul(a, b, out=c)      # no alloc, no RNG -> pure compute
        idle = period - (time.perf_counter() - t0)
        if idle > 0:
            time.sleep(idle)            # yields the core -> lowers thermal lo


class BackgroundProcessManager:
    def __init__(self, num_processes: int, bg_device=None):
        """
        Start `num_processes` worker processes in paused state.
        Computation begins only after calling `resume_processes(delay, duration)`.

        `bg_device` is accepted for backwards compatibility but ignored: the workers
        always contend for the client CPU (see note in `cpu_intensive_task`).
        """
        self.num_processes = num_processes
        self.processes: list[multiprocessing.Process] = []

        print(f"Launching {num_processes} worker processes (initially paused).", self.get_current_time())

        ctx = mp.get_context("spawn")
        # event starts "cleared" → paused
        self.pause_event = ctx.Event()
        self.pause_event.clear()
        p = ctx.Process(
            target=cpu_intensive_task,
            args=(self.pause_event,)
        )
        p.start()
        self.processes.append(p)

    def pause_processes(self) -> None:
        print("Pausing processes...", self.get_current_time())
        self.pause_event.clear()

    def resume_processes(self, delay: float = 0, duration: float | None = None) -> None:
        print("Request to resume processes...", self.get_current_time())
        def delayed_resume():
            if delay > 0:
                time.sleep(delay)
            print("Resuming processes...", self.get_current_time())
            self.pause_event.set()

            if duration is not None:
                time.sleep(duration)
                # only pause if still running
                if self.pause_event.is_set():
                    print("Auto-pausing after duration.", self.get_current_time())
                    self.pause_event.clear()

        threading.Thread(target=delayed_resume, daemon=True).start()

    def terminate_processes(self) -> None:
        print("Terminating processes...", self.get_current_time())
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
        self.processes = []

    def get_current_time(self):
        return datetime.now().strftime("%H:%M:%S")
