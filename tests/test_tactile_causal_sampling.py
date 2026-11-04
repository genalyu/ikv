import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location("tactile_sampling_impl",Path(__file__).parents[1]/"script/encode_tactile_latent.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_tactile_uses_exact_source_stride_and_retains_last_causal_anchor(monkeypatch):
    seen=[]
    def decode(cmd,**kwargs):
        seen.extend(cmd)
        return np.zeros((117,2,2,3),dtype=np.uint8).tobytes()
    monkeypatch.setattr(m.subprocess,"check_output",decode)
    req=SimpleNamespace(length=349,start_timestamp=0,end_timestamp=349/30,source_video=Path("fixture.mp4"))
    frames,ids=m.extract_frames(req,2,2,10,30)
    assert ids==list(range(0,349,3))
    assert len(frames)==117 and ids[-1]==348
    assert any("select=not(mod" in v for v in seen)
    assert "-vsync" in seen

def test_tactile_rejects_fabricated_frame_grid(monkeypatch):
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**k:bytes(116*2*2*3))
    req=SimpleNamespace(length=349,start_timestamp=0,end_timestamp=349/30,source_video=Path("fixture.mp4"))
    with pytest.raises(RuntimeError,match="causal frame grid"):
        m.extract_frames(req,2,2,10,30)
