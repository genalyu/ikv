import numpy as np
import pytest
from n0_twam.task_pipeline.features import estimate_depth_contact

def test_indentation_contact_has_explicit_area_and_depth_thresholds():
    depth=np.full((20,10,10),30.,dtype=np.float32)
    depth[4:8,:2,:2]-=.2
    assert np.flatnonzero(estimate_depth_contact(depth)).tolist()==[4,5,6,7]
    depth[4:8,:2,:2]+= .15
    assert not estimate_depth_contact(depth).any()

def test_static_depth_and_nonfinite_input():
    assert not estimate_depth_contact(np.ones((5,2,2))).any()
    with pytest.raises(ValueError,match="finite"):
        estimate_depth_contact(np.full((3,2,2),np.nan))
