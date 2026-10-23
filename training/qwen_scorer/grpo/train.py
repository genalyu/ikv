"""Run offline final-frame FM-reward GRPO without modifying a live SFT job."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
import sys
import torch
import yaml
from .fm import FMReward, map_scores
from .objective import backward_group


def load_symbol(value):
    module, name = value.split(":")
    return getattr(importlib.import_module(module), name)


def build_reward(config):
    from n0_twam.train import Trainer
    from n0_twam.models.utils import load_mot_checkpoint
    cfg = deepcopy(load_symbol(config["n0_config"]))
    cfg.enable_wandb = False
    cfg.rank = 0
    cfg.local_rank = torch.device(config["n0_device"]).index or 0
    cfg.param_dtype = torch.bfloat16
    cfg.use_ikv_training = True
    cfg.ikv_train_execution = "masked"
    cfg.ikv_train_sample_capacity = False
    cfg.ikv_train_compact_attention = False
    cfg.tactile_cfg_prob = 0.
    cfg.noisy_cond_prob_tactile = 0.
    trainer = Trainer(cfg, inference_only=True)
    trainer.device = torch.device(config["n0_device"])
    model = load_mot_checkpoint(config["n0_checkpoint"], torch_dtype=torch.bfloat16,
                                torch_device=config["n0_device"], attn_mode="flex")
    if getattr(model, "use_local_tactile", False) or getattr(model, "use_contact_gate", False):
        raise ValueError("Initial final-frame reward path requires local tactile/contact gate OFF; "
                         "otherwise current expert tactile could bypass history masking")
    return FMReward(trainer, model, config["reward_weights"], config["reward_scales"])


def validate_record(record):
    for key in ("id","episode_id","task","query_frame","fps","rgb_npy","n0_batch",
                "kv_coordinates","target_raw_frame","alignment_provenance"):
        if key not in record:
            raise ValueError("Missing aligned manifest field: " + key)
    if record["target_raw_frame"] <= record["query_frame"]:
        raise ValueError("Expert target must follow the Qwen observation prefix")
    if not record["alignment_provenance"]:
        raise ValueError("Explicit RGB/latent time provenance required")


def run(config):
    from .policy import QwenPolicy
    from .safety import require_idle_devices
    require_idle_devices(config)
    if config.get("group_size",4) != 4:
        raise ValueError("Initial validated configuration uses exactly four candidates")
    accumulation = int(config.get("gradient_accumulation_frames",4))
    if accumulation < 1:
        raise ValueError("Positive frame accumulation required")
    output = Path(config["output"])
    # New run directories only: cannot overwrite SFT or previous RL results.
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(config,indent=2))
    torch.manual_seed(config.get("seed",42))
    rng = random.Random(config.get("seed",42))
    records = [json.loads(s) for s in Path(config["manifest"]).read_text().splitlines() if s.strip()]
    if not records:
        raise ValueError("Empty manifest")
    for record in records:
        validate_record(record)
    reward_model = build_reward(config)
    policy = QwenPolicy(config)
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.get("learning_rate",1e-6))
    updates = 0
    for epoch in range(config.get("epochs",1)):
        rng.shuffle(records)
        for begin in range(0,len(records),accumulation):
            group_frames = records[begin:begin+accumulation]
            optimizer.zero_grad(set_to_none=True)
            for record in group_frames:
                prompt = policy.prompt(record)
                completions, old, reference = policy.sample(prompt,4)
                budgets = list(rng.choice(config["budget_choices"]))
                raw = torch.load(record["n0_batch"],map_location="cpu",weights_only=True)
                # n0_batch is a batched dictionary from the existing dataset.
                prepared = reward_model.prepare(raw, rng.randrange(2**31), rng.randrange(1000))
                mapping = json.loads(Path(record["kv_coordinates"]).read_text())
                frames = prepared["latent_dict"]["targets"].shape[2]
                patch = reward_model.trainer.patch_size
                shape = prepared["latent_dict"]["targets"].shape
                per_frame = (shape[-2]//patch[1])*(shape[-1]//patch[2])
                if len(mapping) != frames*per_frame:
                    raise ValueError("KV provenance map does not match native dense video layout")
                if any(any(v<0 for v in row) for row in mapping[:-per_frame]):
                    raise ValueError("Every historical video token must have a scored RGB address")
                if any(row != [-1]*4 for row in mapping[-per_frame:]):
                    raise ValueError("Target video tokens must not be mapped to observed RGB")
                rewards, detail, masks, invalid = [], [], [], []
                for completion in completions:
                    try:
                        scores = policy.decode(completion,record)
                    except (ValueError,AssertionError,KeyError,TypeError,json.JSONDecodeError):
                        rewards.append(None);detail.append(None);masks.append(None);invalid.append(True)
                        continue
                    values = map_scores(scores,mapping)
                    r, losses, selected = reward_model.evaluate(prepared,values,budgets)
                    rewards.append(r);detail.append(losses);masks.append(selected);invalid.append(False)
                valid = [r for r in rewards if r is not None]
                penalty = (min(valid) if valid else 0.) - config.get("invalid_margin",1.)
                rewards = [penalty if r is None else r for r in rewards]
                loss = backward_group(policy,prompt,completions,old,reference,rewards,len(group_frames))
                row = dict(record=record["id"],epoch=epoch,update=updates,budgets=budgets,
                           rewards=rewards,fm=detail,invalid=invalid,loss=loss,
                           unique_masks=len({tuple(m) for m in masks if m is not None}))
                with (output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
            norm = torch.nn.utils.clip_grad_norm_(parameters,config.get("max_grad_norm",1.))
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite Qwen gradient")
            optimizer.step()
            updates += 1
            if updates % config.get("save_every",10) == 0:
                save(policy,optimizer,output,updates)
            if updates >= config.get("max_updates",2**31):
                save(policy,optimizer,output,updates,final=True)
                return
    save(policy,optimizer,output,updates,final=True)


def save(policy,optimizer,output,step,final=False):
    path = output/("final" if final else f"checkpoint-{step:06d}")
    policy.save(path)
    torch.save(dict(step=step,optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state()),path/"training_state.pt")
    (path/"COMPLETE.json").write_text(json.dumps(dict(step=step,stage="grpo")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
