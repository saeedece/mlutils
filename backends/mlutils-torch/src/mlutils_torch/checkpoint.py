import re
import time
from concurrent.futures import Future
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, override

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mlutils.utils import GarbageCollector
from mlutils.logging import logger


class ModelContainer(Stateful):
    def __init__(self, model_parts: list[nn.Module]) -> None:
        self.model_parts: list[nn.Module] = model_parts

    @property
    def _state_dict(self) -> dict[str, Any]:
        return {
            k: v
            for sd in map(get_model_state_dict, self.model_parts)
            for k, v in sd.items()
        }

    @override
    def state_dict(self) -> dict[str, Any]:
        return self._state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        f = partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=True),
        )
        _ = list(map(f, self.model_parts))


class OptimizerContainer(Stateful):
    def __init__(self, model_parts: list[nn.Module], optimizers: list[Optimizer]):
        self.model_parts: list[nn.Module] = model_parts
        self.optimizers: list[Optimizer] = optimizers

    @property
    def _state_dict(self):
        f = partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(f, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    @override
    def state_dict(self) -> dict[str, Any]:
        return self._state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        f = partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        _ = list(map(f, self.model_parts, self.optimizers))


class SchedulerContainer(Stateful):
    def __init__(self, schedulers: list[LRScheduler]):
        self.schedulers: list[LRScheduler] = schedulers

    @override
    def state_dict(self) -> dict[str, Any]:
        if len(self.schedulers) > 0:
            return self.schedulers[0].state_dict()
        else:
            return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for scheduler in self.schedulers:
            scheduler.load_state_dict(deepcopy(state_dict))


class CheckpointConfig(NamedTuple):
    folder_path: Path
    interval: int
    model_only: bool


class CheckpointManager:
    def __init__(
        self,
        model: ModelContainer,
        optimizer: OptimizerContainer,
        scheduler: SchedulerContainer,
        config: CheckpointConfig,
    ) -> None:
        # load data from config
        self.folder_path = config.folder_path
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.interval = config.interval

        if config.model_only:
            self.states = {"model": model}
        else:
            self.states = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }

        self.save_future = None
        self.process_group = dist.new_group(backend="gloo")
        return

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: Path,
        enable_garbage_collection: bool = False,
    ) -> Future[None]:
        future = dcp.async_save(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=self.process_group,
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,
        )
        if enable_garbage_collection:
            GarbageCollector.collect("GC collection invoked by checkpointer.")

        return future

    @torch.no_grad()
    def save(self, step: int, last_step: bool = False):
        begin = time.monotonic()
        checkpoint_id = self._create_checkpoint_id(step)
        self._async_wait()

        GarbageCollector.collect("GC collection invoked by checkpointer.")
        self.save_future = self.dcp_save(self.states, checkpoint_id=checkpoint_id)
        GarbageCollector.collect("GC collection invoked by checkpointer.")

        logger.info(
            f"Finished saving checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    @torch.no_grad()
    def load(self, checkpoint_id: Path) -> bool:
        if not checkpoint_id.exists():
            return False

        logger.info(f"Loading checkpoint from {checkpoint_id}")
        begin = time.monotonic()
        states = self._states_to_load()
        dcp.load(states, checkpoint_id=checkpoint_id)
        GarbageCollector.collect("GC collection invoked by checkpointer.")
        logger.info(
            f"Finished loading checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _async_wait(self) -> None:
        if self.save_future is not None:
            self.save_future.result()
            self.save_future = None

    def _create_checkpoint_id(self, step: int) -> Path:
        return self.folder / f"step-{step}"

    def _find_load_step(self) -> int:
        pattern = r"step-(\d+)"

        if not self.folder.exists() or not self.folder.is_dir():
            return -1

        max_step = -1
        for fname in self.folder.iterdir():
            match = re.search(pattern, str(fname))
            if match is not None:
                max_step = max(max_step, int(match.group(1)))

        return max_step

    def _states_to_load(self, model_only: bool = False) -> dict[str, Any]:
        if model_only:
            return self.states["model"].state_dict()

        return self.states
