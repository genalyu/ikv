import numpy as np
import torch
import pytest
from test_dense_kv_index_server import dense_server

@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('history', [False, True])
def test_neoforce_native_rgb_without_mutating_policy(monkeypatch, history, active):
    server = dense_server()
    cfg=server.job_config
    cfg.kv_neoforce_online=True
    cfg.kv_contact_pair_camera_keys=['left']
    cfg.tactile_keys=['a','b']
    server._kv_neoforce_encoder=object()
    server._tactile_image_size=lambda: (64,64)
    server._rgb_motion_streaming_anchor_indices=lambda *a,**k: [0]
    image=np.zeros((4,4,3),dtype=np.uint8);image[...,0]=17;image[...,2]=93
    obs={'obs':[{'left':image}] if history else {'left':image}, 'tactile':{}}
    def build(got,**kw):
        actual=got['obs'][0]['left'] if history else got['obs']['left']
        np.testing.assert_array_equal(actual,image)
        return {'response':torch.ones(1) if active else torch.zeros(1),'visual_rows':torch.tensor([-1])}
    monkeypatch.setattr('n0_twam.preprocessing.online_contact.build_online_contact_pairs', build)
    server._prepare_online_contacts(obs,cold=True)
    assert image[0,0].tolist()==[17,0,93]
