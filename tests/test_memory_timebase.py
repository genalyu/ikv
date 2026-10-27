import numpy as np
import pytest
from training.memory.timebase import action_indices,observation_steps


def test_action_roundtrip_and_chunk_continuity():
    source=np.arange(100)
    native=source[np.arange(120)*25//30]
    pieces=[]
    for start in range(0,120,12):
        pieces.append(native[start:start+12][action_indices(start,12)])
    np.testing.assert_array_equal(np.concatenate(pieces),source)
    assert action_indices(0,12).tolist()==[0,2,3,4,5,6,8,9,10,11]


def test_observations_match_training_causal_rgb_frames():
    steps=[]
    for start in range(0,100,10):steps+=observation_steps(start,10)
    assert steps==[k*25//10 for k in range(1,41)]
    assert observation_steps(0,10)==[2,5,7,10]
    with pytest.raises(ValueError,match="align"):
        action_indices(0,11)
