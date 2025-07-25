import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import jax


@contextmanager
def maybe_enable_profiling(config: dict[str, Any], step: int = 0):
    enable = config["enable"]

    if enable:
        trace_folder = Path(config["trace_folder"])
        trace_folder.mkdir(parents=True, exist_ok=True)
        frequency = config["frequency"]
        yield jax.profiler.trace(log_dir=trace_folder)

    else:
        jax_profiler = nullcontext()
        yield None
