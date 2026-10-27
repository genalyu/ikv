"""Entry point for matched RGB Memory baseline and frozen-scorer training.

Machine paths and experiment choices live in an explicit external JSON. The
upstream LingBot architecture is imported from its recorded source directory;
only dataset, mask selection, optimizer durability and validation are adapted.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from easydict import EasyDict
from .sidecar import verify_binding, sha256


def configure_memory(cfg, spec):
    cfg.memory_plan_root = None
    cfg.memory_task_alias = spec.get("task_alias", {})
    if spec["mode"] == "scorer":
        binding = verify_binding(spec)
        if binding.get("chunk_size") != 2:
            raise ValueError("Scorer plan must match native two-frame serving")
        cfg.memory_plan_root = binding["plan_root"]
        cfg.memory_binding_sha256 = sha256(spec["binding"])
        cfg.memory_scorer_sha256 = binding["scorer_sha256"]
        cfg.memory_budgets = binding["budgets"]
        if cfg.memory_budgets != [3584, 512]:
            raise ValueError("This fixed-budget recipe requires video3584/action512")
    elif spec["mode"] != "baseline":
        raise ValueError("Choose baseline or scorer explicitly")
    cfg.require_selected_episodes = True
    cfg.memory_chunk_size = 2
    cfg.memory_seed_observation = True
    cfg.max_latent_frames = 0
    cfg.batch_size = 1
    cfg.num_init_worker = 1
    cfg.use_rgb_motion_tokens = False
    cfg.ikv_index_root_name = None
    cfg.tactile_keys = []
    cfg.tactile_optional = True
    cfg.use_local_tactile = False
    cfg.tactile_global_zero = True
    cfg.seed = int(spec.get("seed", 42))
    cfg.save_training_state = True
    cfg.task_mode = spec["mode"]
    cfg.memory_experiment = copy.deepcopy(spec)
    selections={str(p):sha256(p) for p in sorted(Path(cfg.dataset_path).glob("*/episode_selection.json"))}
    from .selection import VERSION
    identity=dict(protocol=VERSION,chunk_size=cfg.memory_chunk_size,action_seed="measured-observation-v1",spec=spec,normalization=cfg.norm_stat,cameras=list(cfg.obs_cam_keys),
        action_channels=list(cfg.used_action_channel_ids),episode_selections=selections,
        binding_sha256=getattr(cfg,"memory_binding_sha256",None))
    cfg.task_fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cfg.enable_wandb = False


def memory_input(original, batch, cfg):
    result = original(batch)
    # Match native serving: two latent frames per generation/grounding chunk.
    # Random widths would change the available prefix and current-token budget.
    result["chunk_size"] = cfg.memory_chunk_size
    if cfg.memory_plan_root:
        if "memory_plan" not in batch:
            raise ValueError("Frozen scorer plan is required; no heuristic fallback")
        result["ikv_training"] = dict(capacity=sum(cfg.memory_budgets),
            retention=dict(cfg.kv_retention), execution="masked", block_size=64,
            compact_attention=False, sample_capacity=False,
            causal_memory_plan=batch["memory_plan"])
    return result


def make_n0(spec, rank, local_rank, world):
    from n0_twam.task_pipeline.config import load_task, training_config
    from n0_twam.train import Trainer
    cfg = training_config(load_task(spec["task_config"]), "baseline", world)
    configure_memory(cfg, spec)
    cfg.use_ikv_training = spec["mode"] == "scorer"
    cfg.ikv_train_capacity = 4096
    cfg.ikv_train_execution = "masked"
    cfg.ikv_train_sample_capacity = False
    cfg.ikv_train_min_capacity = 4096
    cfg.kv_retention.update(video_capacity=3584, action_capacity=512, tactile_capacity=0,
        task_weight=0., persistence_weight=0., class_recency_weight=0.)
    cfg.save_root = spec["save_root"]
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world
    cfg.update(spec.get("training", {}))
    if spec.get("resume"):
        cfg.resume_from = str(Path(spec["resume"]).parent)
    class MemoryN0(Trainer):
        def _prepare_input_dict(self, batch):
            return memory_input(super()._prepare_input_dict, batch, self.config)
        def _train_step(self, batch, batch_idx):
            batch = self.convert_input_format(batch)
            inp = self._prepare_input_dict(batch)
            self.transformer.set_requires_gradient_sync(True)
            values = self.compute_loss(inp, self.transformer(inp, train_mode=True))
            finite = torch.tensor(int(torch.isfinite(values["total_loss"])), device=self.device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite loss; abort before optimizer update")
            values["total_loss"].backward()
            values = {k:v.detach() for k,v in values.items()}
            sync = (batch_idx+1) % self.gradient_accumulation_steps == 0
            values["should_log"] = sync
            if sync:
                values["total_norm"] = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2., error_if_nonfinite=True)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
            from .runtime import record_metrics
            record_metrics(self, values, batch_idx)
            return values
        def save_checkpoint(self):
            from .runtime import begin_checkpoint
            begin_checkpoint(self, self.save_dir/f"checkpoint_step_{self.step}")
            super().save_checkpoint()
            if self.config.rank == 0:
                from .sidecar import write_training_manifest
                write_training_manifest(self)
            if dist.is_initialized():
                dist.barrier()
    return MemoryN0(cfg)


def import_lingbot(root):
    directory = Path(root)/"wan_va"
    sys.path.insert(0, str(directory))
    loader = importlib.util.spec_from_file_location("memory_lingbot_upstream_train", directory/"train.py")
    module = importlib.util.module_from_spec(loader)
    # The adapter replaces the upstream dataset entirely. Bind it before the
    # upstream trainer import so unused RobotWin/SciPy preprocessing is not run.
    from n0_twam.dataset import lerobot_latent_dataset as audited_dataset
    sys.modules["dataset"] = audited_dataset
    loader.loader.exec_module(module)
    from n0_twam.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
    module.MultiLatentLeRobotDataset = MultiLatentLeRobotDataset
    return module


def make_lingbot(spec, rank, local_rank, world):
    upstream = import_lingbot(spec["source_root"])
    cfg = copy.deepcopy(upstream.VA_CONFIGS["robotwin_train"])
    cfg.update(spec["model_config"])
    cfg.update(spec.get("training", {}))
    cfg.norm_stat = json.loads(Path(spec["normalization"]).read_text())
    cfg.used_action_channel_ids = [14,15,16,17,18,19,28,21,22,23,24,25,26,29]
    configure_memory(cfg, spec)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world
    cfg.save_root = spec["save_root"]
    cfg.env_type = "none"
    cfg.kv_retention = {}
    if spec.get("resume"):
        cfg.resume_from = str(Path(spec["resume"]).parent)
    from .lingbot import trainer_class
    return trainer_class(upstream)(cfg)


def main():
    import logging
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    spec = json.loads(Path(a.config).read_text())
    if a.resume:
        spec["resume"] = a.resume
    rank, local, world = [int(os.getenv(k, d)) for k,d in (("RANK",0),("LOCAL_RANK",0),("WORLD_SIZE",1))]
    torch.cuda.set_device(local)
    from n0_twam.distributed.util import init_distributed
    init_distributed(world, local, rank)
    seed = int(spec.get("seed",42))+rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Resume path is not part of experiment identity.
    resume = spec.pop("resume", None)
    if spec["model"] not in ("n0twam", "lingbot"):
        raise ValueError("Unknown Memory foundation model")
    builder = make_n0 if spec["model"] == "n0twam" else make_lingbot
    trainer = builder(spec, rank, local, world)
    if resume:
        from n0_twam.task_pipeline.checkpoint import load_training_state
        load_training_state(trainer, resume)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
