from contextlib import contextmanager, nullcontext
from pathlib import Path

import jax


class ProfilingConfig:
    enable: bool
    trace_folder_path: Path
    frequency: int


@contextmanager
def maybe_enable_profiling(config: ProfilingConfig, step: int = 0):
    enable = config.enable

    if enable:
        trace_folder = Path(config.trace_folder_path)
        trace_folder.mkdir(parents=True, exist_ok=True)
        yield jax.profiler.trace(log_dir=trace_folder)

    else:
        jax_profiler = nullcontext()
        yield None
