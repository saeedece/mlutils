from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from matplotlib.figure import Figure
from torch.utils.tensorboard.writer import SummaryWriter
from torch._utils import _get_available_device_type, _get_device_module  # pyright: ignore[reportPrivateUsage]

from mlutils.metrics import DeviceStats


class TensorboardLoggerConfig(NamedTuple):
    enable_tensorboard: bool


class TensorboardLogger:
    def __init__(
        self,
        config: TensorboardLoggerConfig,
        log_path: Path,
        tag: str | None = None,
    ) -> None:
        self.tag = tag
        self.writer = (
            SummaryWriter(log_path, max_queue=1000)
            if config.enable_tensorboard
            else None
        )

    def log_scalars(self, tag_scalar_dict: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return

        self.writer.add_scalars(
            main_tag=self.tag or "",
            tag_scalar_dict=tag_scalar_dict,
            global_step=step,
        )

    def log_images(
        self,
        img_tensor: npt.NDArray[np.integer],
        img_tag: str,
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        if self.writer is None:
            return

        img_tag = f"{self.tag}/{img_tag}" if self.tag else img_tag
        self.writer.add_images(
            tag=img_tag,
            img_tensor=img_tensor,
            global_step=step,
            dataformats=dataformats,
        )

    def log_figures(self, figures: list[Figure], figure_tag: str, step: int) -> None:
        if self.writer is None:
            return

        figure_tag = f"{self.tag}/{figure_tag}" if self.tag else figure_tag
        self.writer.add_figure(
            tag=figure_tag,
            figure=figures,
            global_step=step,
            close=True,
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class TorchMonitor:
    device_module = _get_device_module(_get_available_device_type() or "cuda")

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.device_name = TorchMonitor.device_module.get_device_name(self.device)
        self.device_index = TorchMonitor.device_module.current_device()
        self.device_capacity = TorchMonitor.device_module.get_device_properties(
            self.device
        ).total_memory

        TorchMonitor.device_module.reset_peak_memory_stats()
        TorchMonitor.device_module.empty_cache()

    def get_stats(self) -> DeviceStats:
        device_info = TorchMonitor.device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = max_active / (1024**3)
        max_active_pct = 100 * max_active / self.device_capacity

        max_reserv = device_info.get("reserved_bytes.all.peak", -1)
        max_reserv_gib = max_reserv / (1024**3)
        max_reserv_pct = 100 * max_reserv / self.device_capacity

        num_alloc_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        return DeviceStats(
            max_active_gib,
            max_active_pct,
            max_reserv_gib,
            max_reserv_pct,
            num_alloc_retries,
            num_ooms,
        )
