import gc
import time

from mlutils.logging import logger


class GarbageCollector:
    def __init__(self, interval: int = 1000):
        self.interval: float = interval
        gc.disable()

    def run(self, step: int) -> None:
        if step % self.interval == 0:
            self.collect("Periodic garbage collection.")

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        _ = gc.collect(generation)
        logger.info(f"GC: {reason} {time.monotonic() - begin:.2f} seconds.")
