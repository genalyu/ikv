"""LingBot architecture adapter with explicit Memory masks and durable training."""
import copy
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from safetensors.torch import save_file
from n0_twam.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from n0_twam.task_pipeline.checkpoint import save_training_state, rng_state, restore_rng
from .selection import support_from_plan


def install_mask_adapter():
    from modules.model import FlexAttnFunc
    if hasattr(FlexAttnFunc, "memory_original_init"):
        return FlexAttnFunc
    original = FlexAttnFunc.init_mask
    FlexAttnFunc.memory_original_init = original
    FlexAttnFunc.memory_selected = None
    def init(latent_shape, action_shape, padded_length, chunk_size, window_size, patch_size, device):
        original(latent_shape, action_shape, padded_length, chunk_size, window_size, patch_size, device)
        selected = FlexAttnFunc.memory_selected
        if selected is None:
            return
        B, _, frames, H, W = latent_shape
        if B != 1 or chunk_size != 2 or tuple(patch_size) != (1,2,2):
            raise ValueError("LingBot Memory requires B1/chunk2/native patches")
        v = H//2*(W//2); a = action_shape[3]*action_shape[4]
        V, A = frames*v, frames*a
        vf = (torch.arange(frames, device=device)//chunk_size).repeat_interleave(v)*2
        af = (torch.arange(frames, device=device)//chunk_size).repeat_interleave(a)*2+1
        phase = torch.cat([vf,vf,af,af,torch.full((padded_length,),-1,device=device)])
        clean = torch.cat([torch.zeros(V,device=device),torch.ones(V,device=device),
            torch.zeros(A,device=device),torch.ones(A,device=device),torch.zeros(padded_length,device=device)]).bool()
        kind = torch.cat([torch.zeros(2*V,device=device),torch.ones(2*A,device=device),
            torch.zeros(padded_length,device=device)]).long()
        seq = torch.where(phase>=0,0,-1)
        plan = support_from_plan(dict(seq=seq,phase=phase,clean=clean,kind=kind), selected, selected.shape[-1])["clean"]
        def mask(b,h,q,k):
            valid = (seq[q]>=0)&(seq[k]>=0)
            same = (phase[q]==phase[k])&(clean[q]==clean[k])
            prior = (phase[k]<phase[q])&clean[k]&plan[phase[q].clamp_min(0),k]
            return valid&(same|prior)
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask,1,1,len(seq),len(seq),device=device,BLOCK_SIZE=64,_compile=True)
    FlexAttnFunc.init_mask = staticmethod(init)
    return FlexAttnFunc


def trainer_class(upstream):
    class MemoryLingbot(upstream.Trainer):
        def __init__(self, config):
            super().__init__(config)
            self.data_epoch = self.data_offset = 0
            self.mask_class = install_mask_adapter()
            # Single-GPU runs also need deterministic sampler state for resume.
            sampler = DistributedSampler(self.train_loader.dataset, num_replicas=config.world_size,
                rank=config.rank, seed=config.seed, shuffle=True)
            self.train_loader = DataLoader(self.train_loader.dataset, batch_size=1,
                sampler=sampler, num_workers=config.load_worker)
            self.val_loader = None
            if getattr(config, "val_dataset_path", None):
                vc = copy.deepcopy(config); vc.dataset_path=vc.val_dataset_path; vc.cfg_prob=0.
                dataset = MultiLatentLeRobotDataset(vc)
                # No DistributedSampler padding/duplicates in validation metrics.
                indices = list(range(config.rank,len(dataset),config.world_size))
                self.val_loader = DataLoader(torch.utils.data.Subset(dataset, indices),batch_size=1,num_workers=0)

        def _get_next_batch(self):
            if self.train_loader_iter is None:
                self.train_loader.sampler.set_epoch(self.data_epoch)
                self.train_loader_iter=iter(self.train_loader)
            try:
                value=next(self.train_loader_iter)
            except StopIteration:
                self.data_epoch+=1; self.data_offset=0
                self.train_loader.sampler.set_epoch(self.data_epoch)
                self.train_loader_iter=iter(self.train_loader)
                value=next(self.train_loader_iter)
            self.data_offset+=1
            return value

        def _prepare_input_dict(self,batch):
            result=super()._prepare_input_dict(batch)
            result["chunk_size"]=self.config.memory_chunk_size
            self.mask_class.memory_selected=batch.get("memory_plan")
            if self.config.memory_plan_root and self.mask_class.memory_selected is None:
                raise ValueError("Missing scorer plan")
            return result

        def _train_step(self,batch,batch_idx):
            batch=self.convert_input_format(batch)
            inp=self._prepare_input_dict(batch)
            # FSDP2 reduce every microbatch; delayed reduction can mix sharded and
            # full gradients. Loss/accumulation scaling remains unchanged.
            self.transformer.set_requires_gradient_sync(True)
            out=self.transformer(inp,train_mode=True)
            lv,la=self.compute_loss(inp,out)
            loss=lv+la
            finite=torch.tensor(int(torch.isfinite(loss)),device=self.device)
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite loss; abort before optimizer update")
            loss.backward()
            sync=(batch_idx+1)%self.gradient_accumulation_steps==0
            values=dict(latent_loss=lv.detach(),action_loss=la.detach(),should_log=sync)
            if sync:
                norm=torch.nn.utils.clip_grad_norm_(self.transformer.parameters(),2.,error_if_nonfinite=True)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
                values["total_norm"]=norm
            from .runtime import record_metrics
            record_metrics(self, values, batch_idx)
            return values

        @torch.no_grad()
        def validate(self):
            if self.val_loader is None:
                return
            saved=rng_state(); self.transformer.eval()
            try:
                torch.manual_seed(7721+self.config.rank)
                metrics=torch.zeros(3,device=self.device,dtype=torch.float64)
                # FSDP ranks need identical numbers of forwards. Memory corpus
                # has 60 episodes; require divisibility rather than duplicate.
                counts=torch.tensor(len(self.val_loader),device=self.device)
                lo=counts.clone();hi=counts.clone()
                dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
                if lo!=hi:
                    raise ValueError("Validation corpus must divide over GPU count")
                for batch in self.val_loader:
                    inp=self._prepare_input_dict(self.convert_input_format(batch))
                    lv,la=self.compute_loss(inp,self.transformer(inp,train_mode=True))
                    metrics+=torch.stack([lv.double()*self.gradient_accumulation_steps,
                        la.double()*self.gradient_accumulation_steps,lv.new_tensor(1).double()])
                dist.all_reduce(metrics)
                if self.config.rank==0:
                    p=Path(self.config.save_root)/"validation.jsonl"
                    with p.open("a") as f:
                        f.write(json.dumps(dict(step=self.step,episodes=int(metrics[2]),
                            latent_loss=float(metrics[0]/metrics[2]),action_loss=float(metrics[1]/metrics[2])))+"\n")
            finally:
                restore_rng(saved); self.transformer.train()

        def save_checkpoint(self):
            directory=self.save_dir/f"checkpoint_step_{self.step}"
            from .runtime import begin_checkpoint
            begin_checkpoint(self, directory)
            from .export import export_sharded
            export_sharded(self.transformer,directory/"transformer")
            save_training_state(self,directory/"training_state")
            if self.config.rank==0:
                from .sidecar import write_training_manifest
                write_training_manifest(self)
            if self.step%int(getattr(self.config,"val_interval",100))==0:
                self.validate()
    return MemoryLingbot
