import h5py
import numpy as np
import pytest
from n0_twam.task_pipeline.neosim import iter_neosim

def make_task(tmp_path, arms):
    (tmp_path/"hdf5").mkdir()
    with h5py.File(tmp_path/"hdf5/0.hdf5","w") as h:
        h["step"]=np.arange(6)*4
        for i,name in enumerate(["embodiment","embodiment_b"][:arms]):
            ee=np.zeros((6,7));ee[:,0]=i+.1+np.arange(6)*.01;ee[:,3]=1
            h[name+"/ee"]=ee
            j=np.zeros((6,8));j[:,6:]=.039;h[name+"/joint"]=j
        h["observation/head/rgb"]=np.zeros((6,),dtype="S1")
        h["tactile/left_tactile/rgb"]=np.zeros((6,),dtype="S1")
    return dict(source=str(tmp_path),action_labels="next_observation",robot=dict(
        arms=arms,embodiment_paths=["embodiment","embodiment_b"][:arms],
        quaternion_order="wxyz",simulation_hz=120,fps=30,
        gripper_joint_indices=[6,7],gripper_stroke_m=.078,
        cameras={"observation/head/rgb":"observation.images.top"},
        tactile={"tactile/left_tactile/rgb":"observation.images.tactile_a"}))

@pytest.mark.parametrize("arms",[1,2])
def test_explicit_arm_mapping_and_next_state_targets(tmp_path,arms):
    task=make_task(tmp_path,arms)
    ep=next(iter_neosim(task))
    assert ep.state.shape==(5,10*arms)
    assert ep.action.shape==ep.state.shape
    for i in range(arms):
        np.testing.assert_allclose(ep.state[:,i*10],i+.1+np.arange(5)*.01)
        np.testing.assert_allclose(ep.action[:,i*10],i+.11+np.arange(5)*.01)
        np.testing.assert_allclose(ep.state[:,i*10+9],1)
    assert ep.metadata["embodiment_paths"]==task["robot"]["embodiment_paths"]

def test_dual_arm_requires_complete_unique_mapping(tmp_path):
    task=make_task(tmp_path,2)
    task["robot"]["embodiment_paths"]=["embodiment","embodiment"]
    with pytest.raises(ValueError,match="every arm exactly once"):next(iter_neosim(task))
