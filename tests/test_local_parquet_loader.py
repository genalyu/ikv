from pathlib import Path
from types import SimpleNamespace
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from n0_twam.dataset.lerobot_latent_dataset import LatentLeRobotDataset

def test_local_loader_preserves_rows_and_torch_transform_without_builder(tmp_path,monkeypatch):
    for i in range(2):
        pq.write_table(pa.table({"episode_index":[i,i],"frame_index":[0,1],
            "action":[[float(i),2.],[float(i),3.]]}),tmp_path/f"ep{i}.parquet")
    def forbidden(*a,**kw):raise AssertionError("HF builder must not run")
    monkeypatch.setattr(datasets,"load_dataset",forbidden)
    fake=SimpleNamespace(root=tmp_path,get_episodes_file_paths=lambda:["ep0.parquet","ep1.parquet"])
    ds=LatentLeRobotDataset.load_hf_dataset(fake)
    assert len(ds)==4
    assert [int(v) for v in ds["episode_index"]]==[0,0,1,1]
    torch.testing.assert_close(ds[2]["action"],torch.tensor([1.,2.]))

def test_repo_discovery_prunes_episode_sidecar_trees(tmp_path,monkeypatch):
    import n0_twam.dataset.lerobot_latent_dataset as m
    repo=tmp_path/"train"/"task";(repo/"meta").mkdir(parents=True)
    (repo/"meta/info.json").write_text("{}")
    (repo/"tactile_index_sources"/"0"/"meta").mkdir(parents=True)
    (repo/"tactile_index_sources"/"0"/"meta/info.json").write_text("{}")
    visited=[]
    original=m.os.walk
    def walk(*a,**kw):
        for row in original(*a,**kw):
            visited.append(row[0]);yield row
    monkeypatch.setattr(m.os,"walk",walk)
    assert m.recursive_find_file(tmp_path)==[str(repo/"meta/info.json")]
    assert not any("tactile_index_sources" in str(x) for x in visited)
