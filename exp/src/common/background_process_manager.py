import time
import threading
import multiprocessing
import multiprocessing.synchronize
from datetime import datetime

import torch


def cpu_intensive_task(
    pause_event: multiprocessing.synchronize.Event,
    device: torch.device
):
    random_tensor = torch.randn(100, 100).to(device)
    print(f"Starting process on {device}")
    while True:
        # wait until "running" (pause_event is set)
        pause_event.wait()
        _ = random_tensor @ random_tensor  # heavy compute


class BackgroundProcessManager:
    def __init__(self, num_processes: int, bg_device: torch.device):
        """
        Start `num_processes` worker processes in paused state.
        Computation begins only after calling `resume_processes(delay, duration)`.
        """
        self.num_processes = num_processes
        self.processes: list[multiprocessing.Process] = []

        # event starts "cleared" → paused
        self.pause_event = multiprocessing.Event()
        self.pause_event.clear()

        print(f"Launching {num_processes} worker processes (initially paused).", self.get_current_time())

        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=cpu_intensive_task,
                args=(self.pause_event, bg_device)
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
