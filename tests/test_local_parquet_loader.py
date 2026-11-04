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
