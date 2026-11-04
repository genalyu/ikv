import errno
from types import SimpleNamespace
import pytest
import n0_twam.dataset.lerobot_latent_dataset as m

def test_transient_eio_retries_same_repo(monkeypatch):
    calls=[]
    expected=object()
    def create(**kw):
        calls.append(kw)
        if len(calls)<3:raise OSError(errno.EIO,"temporary")
        return expected
    monkeypatch.setattr(m,"LatentLeRobotDataset",create)
    monkeypatch.setattr("time.sleep",lambda _:None)
    cfg=SimpleNamespace()
    assert m.construct_lerobot("repo",cfg) is expected
    assert len(calls)==3 and all(x==dict(repo_id="repo",config=cfg) for x in calls)

def test_persistent_eio_is_bounded_and_non_io_errors_do_not_retry(monkeypatch):
    calls=[]
    def create(**kw):
        calls.append(kw);raise OSError(errno.EIO,"persistent")
    monkeypatch.setattr(m,"LatentLeRobotDataset",create)
    monkeypatch.setattr("time.sleep",lambda _:None)
    assert m.construct_lerobot("repo",None) is None
    assert len(calls)==5
    calls.clear()
    def invalid(**kw):
        calls.append(kw);raise ValueError("invalid")
    monkeypatch.setattr(m,"LatentLeRobotDataset",invalid)
    assert m.construct_lerobot("repo",None) is None and len(calls)==1
