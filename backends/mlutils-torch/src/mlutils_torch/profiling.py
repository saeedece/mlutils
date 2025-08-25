import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.distributed as dist

from mlutils.logging import logger


class ProfilingConfig:
    enable: bool
    trace_folder_path: Path
    frequency: int


@contextmanager
def maybe_enable_profiling(config: ProfilingConfig, step: int = 0):
    enable = config.enable
    if enable:
        trace_folder = config.trace_folder_path
        trace_folder.mkdir(parents=True, exist_ok=True)
        frequency = config.frequency

        rank = dist.get_rank()

        def trace_handler(prof):
            curr_trace_folder = trace_folder / f"iteration_{prof.step_num}"
            curr_trace_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            prof.export_chrome_trace(curr_trace_folder / f"rank{rank}_trace.json")
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin} seconds."
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_folder}")
        warmup, active = 3, 1
        wait = frequency - (active + warmup)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = step
            yield torch_profiler

    else:
        torch_profiler = nullcontext()
        yield None
