"""Full optimizer/model resume; distributed shards plus rank-local RNG/loader state."""

import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


class TrainingState:
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        model, optim = get_state_dict(self.trainer.transformer, self.trainer.optimizer)
        return {"model": model, "optimizer": optim}

    def load_state_dict(self, state):
        set_state_dict(
            self.trainer.transformer,
            self.trainer.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(trainer, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(trainer.config.rank)
    world = int(trainer.config.world_size)
    # All ranks participate; DCP saves original parameter precision as well as optimizer moments.
    dcp.save(
        {"training": TrainingState(trainer)}, checkpoint_id=directory / "distributed"
    )
    state = dict(
        step=trainer.step,
        scheduler=trainer.lr_scheduler.state_dict(),
        rng=rng_state(),
        data_epoch=getattr(trainer, "data_epoch", 0),
        data_offset=getattr(trainer, "data_offset", 0),
    )
    torch.save(state, directory / f"rank_{rank}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        manifest = dict(
            world_size=world,
            task_fingerprint=trainer.config.task_fingerprint,
            mode=trainer.config.task_mode,
            ikv_train_execution=getattr(trainer.config, "ikv_train_execution", "recurrent"),
            ikv_train_sample_capacity=bool(getattr(trainer.config, "ikv_train_sample_capacity", False)),
            ikv_train_min_capacity=int(getattr(trainer.config, "ikv_train_min_capacity", 0)),
            step=trainer.step,
        )
        temp = directory / "complete.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(directory / "complete.json")
    if dist.is_initialized():
        dist.barrier()


def load_training_state(trainer, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "complete.json").read_text())
    expected = dict(
        world_size=int(trainer.config.world_size),
        task_fingerprint=trainer.config.task_fingerprint,
        mode=trainer.config.task_mode,
        ikv_train_execution=getattr(trainer.config, "ikv_train_execution", "recurrent"),
        ikv_train_sample_capacity=bool(getattr(trainer.config, "ikv_train_sample_capacity", False)),
        ikv_train_min_capacity=int(getattr(trainer.config, "ikv_train_min_capacity", 0)),
    )
    legacy_defaults = {"ikv_train_execution": "recurrent",
                       "ikv_train_sample_capacity": False,
                       "ikv_train_min_capacity": 0}
    for k, v in expected.items():
        if manifest.get(k, legacy_defaults.get(k)) != v:
            raise ValueError(
                f"Resume {k} mismatch; exact resume requires same task, mode, GPU count and IKV training settings"
            )
    dcp.load(
        {"training": TrainingState(trainer)}, checkpoint_id=directory / "distributed"
    )
    state = torch.load(
        directory / f"rank_{trainer.config.rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    trainer.lr_scheduler.load_state_dict(state["scheduler"])
    trainer.step = state["step"]
    trainer.data_epoch = int(state["data_epoch"])
    trainer.data_offset = 0
    sampler = trainer.train_loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(trainer.data_epoch)
    trainer.train_loader_iter = iter(trainer.train_loader)
    for _ in range(state["data_offset"]):
        next(trainer.train_loader_iter)
        trainer.data_offset += 1
    restore_rng(state["rng"])
