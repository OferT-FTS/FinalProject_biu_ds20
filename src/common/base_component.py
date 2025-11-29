import time
import psutil
import os
import logging
from typing import Any

class BaseComponent:
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("ml_project")

    def timer(self, label: str = "Task"):
        return self._Timer(self.logger, label)

    class _Timer:
        def __init__(self, logger, label):
            self.logger = logger
            self.label = label
            self.process = psutil.Process(os.getpid())

        def __enter__(self):
            self.start_time = time.time()
            self.start_cpu = self.process.cpu_percent(interval=None)
            self.start_mem = self.process.memory_info().rss  # bytes

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            end_cpu = self.process.cpu_percent(interval=None)
            end_mem = self.process.memory_info().rss

            elapsed = end_time - self.start_time
            mem_used = (end_mem - self.start_mem) / (1024 * 1024)  # MB
            cpu_delta = end_cpu - self.start_cpu

            self.logger.info(
                f"[{self.label}] Completed in {elapsed:.2f}s | "
                f"CPU Change: {cpu_delta:.2f}% | "
                f"Memory Change: {mem_used:.2f} MB"
            )

'''How to use:
class DataProcessor(BaseComponent):
    def transform(self, df):
        with self.timer("Feature Engineering"):
            df["new_feature"] = df["value"] * 2
            time.sleep(1.5)   # simulate work
        return df
'''