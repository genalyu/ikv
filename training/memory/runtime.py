"""Runtime acceptance evidence and checkpoint completion discipline."""
import json
from pathlib import Path
import torch
import torch.distributed as dist

def begin_checkpoint(trainer, directory):
    """An interrupted overwrite must never retain an old completion marker."""
    if trainer.config.rank == 0:
        (Path(directory)/"training_state/complete.json").unlink(missing_ok=True)
    if dist.is_initialized():
        dist.barrier()

def record_metrics(trainer, values, batch_idx):
    if trainer.config.rank != 0:
        return
    row=dict(step=trainer.step+1,microbatch=batch_idx,rank=0,
             learning_rate=trainer.lr_scheduler.get_last_lr()[0])
    for key in ("latent_loss","action_loss","total_norm"):
        if key in values:
            row[key]=float(values[key].detach().item())
    path=Path(trainer.config.save_root)/"rank0_training_metrics.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(row,allow_nan=False)+"\n")
