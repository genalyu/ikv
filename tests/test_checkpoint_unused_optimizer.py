from types import SimpleNamespace
import torch
from torch.distributed import checkpoint as dcp
from n0_twam.task_pipeline.checkpoint import TrainingState


def test_unused_optimizer_state_roundtrip_next_update(tmp_path):
    torch.manual_seed(3)
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    model[0](torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    obj=SimpleNamespace(transformer=model,optimizer=optimizer)
    dcp.save({"training":TrainingState(obj)},checkpoint_id=tmp_path)
    restored=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,2))
    opt2=torch.optim.AdamW(restored.parameters(),lr=.01)
    obj2=SimpleNamespace(transformer=restored,optimizer=opt2)
    reader=dcp.FileSystemReader(tmp_path);meta=reader.read_metadata()
    keys={p[3] for p in meta.planner_data.values() if len(p)>=5 and tuple(p[:3])==("training","optimizer","state")}
    dcp.load({"training":TrainingState(obj2,keys)},storage_reader=reader)
    assert not opt2.state[restored[1].weight]
    # Previously unused layer now receives its first gradient. Adam bias
    # correction must start at step 1 on BOTH the uninterrupted/resumed runs.
    for m,o in ((model,optimizer),(restored,opt2)):
        m(torch.ones(1,2)).sum().backward();o.step()
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
