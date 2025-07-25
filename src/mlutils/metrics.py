from pathlib import Path
from typing import Any, NamedTuple


class DeviceStats(NamedTuple):
    max_active_gib: float
    max_active_pct: float
    max_reserv_gib: float
    max_reserv_pct: float
    num_alloc_retries: int
    num_ooms: int
