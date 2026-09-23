import numpy as np
import pytest
import torch
from n0_twam.preprocessing.online_contact import build_online_contact_pairs


class Encoder:
    def __init__(self):
        self.clips = []

    def __call__(self, rgb, force, mask):
        self.clips.append(force.clone())
        b, t = rgb.shape[:2]
        tokens = torch.eye(4).reshape(1, 1, 2, 2, 4).expand(b, t, -1, -1, -1)
        return {'tokens': tokens, 'visual_tokens': tokens,
                'response': force.reshape(b, t, 2, 3, 2, 2).abs().amax(3)}


@pytest.mark.parametrize('sensors', [2, 4])
def test_online_pipeline_sensor_time_camera_layout(sensors):
    keys = [str(i) for i in range(sensors)]
    frames, tactile = [], []
    for t in range(8):
        frames.append({'top': np.zeros((8, 8, 3), dtype=np.uint8)})
        force = np.zeros((sensors, 2, 2, 3), dtype=np.float32)
        force[:, 0, 1, 0] = t + 1
        tactile.append({'__ikv_force_schema__': 'neosim_sensor_xyz_v1',
                        '__ikv_force_keys__': keys, '__ikv_force__': force})
    encoder = Encoder()
    pairs = build_online_contact_pairs(
        {'obs': frames, 'tactile': tactile}, encoder=encoder, anchors=[3, 7],
        camera_keys=['wrist', 'top'], tactile_keys=keys, visual_grid=(2, 2),
        tactile_grid=(2, 2), paired_camera='top', device='cpu')
    assert pairs['neoforce'].shape == (sensors * 2 * 4, 4)
    rows = pairs['visual_rows'].reshape(sensors, 2, 4)
    assert rows[:, :, 1].tolist() == [[3, 11]] * sensors
    assert (rows[:, :, [0, 2, 3]] == -1).all()
    assert [len(x[0]) for x in encoder.clips] == [4] * sensors
    assert encoder.clips[0].abs().max() == 4  # no future leakage
    assert encoder.clips[1].abs().max() == 8


def test_online_missing_force_fails_loudly():
    with pytest.raises(ValueError, match='schema'):
        build_online_contact_pairs(
            {'obs': {'top': np.zeros((2, 2, 3), dtype=np.uint8)}, 'tactile': {}},
            encoder=Encoder(), anchors=[0], camera_keys=['top'],
            tactile_keys=['l', 'r'], visual_grid=(2, 2), tactile_grid=(2, 2),
            paired_camera='top', device='cpu')
