"""Episode feature reuse and exact optimizer-boundary resume."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from training.parallel_scorer.episode_cache import episode_packet
from training.parallel_scorer.model import ScorerConfig

def test_episode_cache_exact_bfloat_prefix_and_revision(tmp_path):
    values=torch.randn(384,12).bfloat16()
    mm=np.memmap(tmp_path/"features.bf16",mode="w+",dtype=np.uint16,shape=(384,12))
    mm[:]=values.view(torch.uint16).numpy();mm.flush();del mm
    torch.save(torch.randn(3,10),tmp_path/"text.pt")
    (tmp_path/"COMPLETE.json").write_text(json.dumps(dict(episode_id="a",task="pick",
        frames=2,fps=25.,vision_dim=12,text_dim=10,encoder_revision="r")))
    record=dict(episode_id="a",task="pick",fps=25.,query_frame=0,
                episode_features=str(tmp_path),target=json.dumps(dict(default_score=50,blocks=[])))
    packet=episode_packet(record,"r")
    torch.testing.assert_close(packet["features"],values[:192],rtol=0,atol=0)
    assert packet["features"].shape==(192,12) and packet["times"].max()==0
    assert packet["targets"].eq(50).all()
    with pytest.raises(ValueError,match="Stale"):episode_packet(record,"wrong")
    with pytest.raises(ValueError,match="prefix"):episode_packet(dict(record,query_frame=2),"r")

def test_exact_resumption_matches_uninterrupted(monkeypatch,tmp_path):
    from training.parallel_scorer import train
    torch.manual_seed(19)
    rows=[]
    for j in range(4):
        packet=dict(features=torch.randn(7,12),text=torch.randn(3,10),
            times=torch.arange(7).float(),xy=torch.rand(7,2),modalities=torch.arange(7)%2,
            sensors=torch.zeros(7,dtype=torch.long),query_time=6.,
            ids=[str(i) for i in range(7)],encoder_revision="test",
            targets=torch.linspace(0,100,7))
        path=tmp_path/f"packet{j}.pt";torch.save(packet,path)
        rows.append(dict(id=str(j),episode_id=str(j),feature_path=str(path)))
    trainfile=tmp_path/"train.jsonl";valfile=tmp_path/"val.jsonl"
    trainfile.write_text("".join(json.dumps(r)+"\n" for r in rows[:3]))
    valfile.write_text(json.dumps(rows[-1])+"\n")
    def preset(name,**kw):
        return ScorerConfig(**dict(kw,width=16,layers=2,heads=2,checkpoint_layers=False))
    monkeypatch.setattr(train.ScorerConfig,"preset",preset)
    def run(output,steps,resume=False):
        argv=["train","--manifest",str(trainfile),"--validation",str(valfile),
          "--output",str(tmp_path/output),"--device","cpu","--revision","test",
          "--epochs","1","--accumulate","1","--checkpoint-steps","1",
          "--cpu-threads","2","--max-updates",str(steps)]
        if resume:argv.append("--resume")
        monkeypatch.setattr(sys,"argv",argv);train.main()
    run("whole",3);run("split",1);run("split",3,True)
    a=torch.load(tmp_path/"whole/last-state.pt",weights_only=True)
    b=torch.load(tmp_path/"split/last-state.pt",weights_only=True)
    assert a["updates"]==b["updates"]==3 and a["cursor"]==b["cursor"]==3
    for k in a["model"]:torch.testing.assert_close(a["model"][k],b["model"][k],rtol=0,atol=0)
    for key,state in a["optimizer"]["state"].items():
        for k,v in state.items():torch.testing.assert_close(v,b["optimizer"]["state"][key][k],rtol=0,atol=0)
    # Resume at epoch boundary performs validation and marks completion.
    run("split",4,True)
    assert (tmp_path/"split/COMPLETE.json").exists()


def test_linear_patch_projection_matches_float32_conv_and_gradients():
    from copy import deepcopy
    from types import SimpleNamespace
    from training.qwen_scorer.vision_patch_linear import enable_linear_patch_embed
    from torch import nn
    class Patch(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Conv3d(3,16,kernel_size=(2,4,4),stride=(2,4,4))
        def forward(self,x):return self.proj(x.reshape(-1,3,2,4,4)).reshape(-1,16)
    a=Patch();b=deepcopy(a);enable_linear_patch_embed(SimpleNamespace(patch_embed=b))
    x=torch.randn(32,96);y=a(x);z=b(x)
    torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-5)
    y.square().sum().backward();z.square().sum().backward()
    for v,w in zip(a.parameters(),b.parameters()):torch.testing.assert_close(v.grad,w.grad,atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize("name,width,heads,count", [("p067m",64,1,66752),("p26m",128,2,264576),("p59m",192,3,593472)])
def test_submillion_scorers(name,width,heads,count):
    from training.parallel_scorer.model import ParallelScorer
    c=ScorerConfig.preset(name,vision_dim=12,text_dim=10,checkpoint_layers=False)
    m=ParallelScorer(c)
    assert (c.width,c.heads,c.layers)==(width,heads,1)
    assert m.counts()["interaction"]==count
    y=m(torch.randn(9,12),torch.randn(3,10),torch.arange(9).float(),torch.rand(9,2),torch.zeros(9,dtype=torch.long),torch.zeros(9,dtype=torch.long),8.)
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
