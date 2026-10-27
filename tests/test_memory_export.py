import json
import torch
from safetensors.torch import load_file
from training.memory.export import export_sharded


def test_sharded_export_roundtrip(tmp_path):
    model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Linear(8,2))
    model.config={"example":True}
    export_sharded(model,tmp_path,max_shard_bytes=64)
    index=json.loads((tmp_path/"diffusion_pytorch_model.safetensors.index.json").read_text())
    assert set(index["weight_map"])==set(model.state_dict())
    loaded={}
    for file in set(index["weight_map"].values()):loaded.update(load_file(str(tmp_path/file)))
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.bfloat16(),loaded[name],rtol=0,atol=0)
    assert len(set(index["weight_map"].values()))>1
