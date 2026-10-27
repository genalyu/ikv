"""Bounded-host-memory inference weight export for FSDP2 checkpoints."""
import gc
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from safetensors.torch import save_file


@torch.no_grad()
def export_sharded(model, directory, max_shard_bytes=512*1024**2):
    directory=Path(directory)
    distributed=dist.is_available() and dist.is_initialized()
    rank=dist.get_rank() if distributed else 0
    state=get_model_state_dict(model,options=StateDictOptions(full_state_dict=False))
    buffer={};weight_map={};size=total=0;shard=0
    if rank==0:directory.mkdir(parents=True,exist_ok=True)
    def flush(buffer,shard):
        error=[None]
        if rank==0:
            try:
                file=f"diffusion_pytorch_model-{shard:05d}.safetensors"
                save_file(buffer,str(directory/file),metadata={"format":"pt"})
                weight_map.update({k:file for k in buffer})
            except Exception as exc:error[0]=repr(exc)
        if distributed:dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
    # All ranks call full_tensor in identical order. At most one native
    # parameter is gathered, rather than the entire FP32 transformer.
    for name,value in state.items():
        full=value.full_tensor() if hasattr(value,"full_tensor") else value
        dtype=torch.bfloat16 if full.is_floating_point() else full.dtype
        nbytes=full.numel()*torch.empty((),dtype=dtype).element_size()
        if size and size+nbytes>max_shard_bytes:
            shard+=1;flush(buffer,shard);buffer={};size=0;gc.collect()
        if rank==0:buffer[name]=full.detach().to(device="cpu",dtype=dtype).contiguous()
        size+=nbytes;total+=nbytes
        del full
    if size:
        shard+=1;flush(buffer,shard)
    error=[None]
    if rank==0:
        try:
            index=dict(metadata={"total_size":total},weight_map=weight_map)
            path=directory/"diffusion_pytorch_model.safetensors.index.json"
            tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(index,indent=2));tmp.replace(path)
            (directory/"config.json").write_text(json.dumps(dict(model.config),indent=2))
        except Exception as exc:error[0]=repr(exc)
    if distributed:dist.broadcast_object_list(error,src=0)
    if error[0]:raise RuntimeError(error[0])
    del state,buffer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
