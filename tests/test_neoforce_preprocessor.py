from types import SimpleNamespace
import pytest
import torch
from n0_twam.preprocessing.neoforce import FrozenNeoForceEncoder
from n0_twam.preprocessing.neoforce import pool_contact_tokens
from n0_twam.preprocessing.neoforce import encode_causal_contact_frames


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(image_size=(4, 4), patch_size=2, temporal_max_frames=4)
        self.received = None

    def forward(self, rgb, **kwargs):
        self.received = kwargs
        return {'patch_t': self.weight * torch.ones(*rgb.shape[:2], 4, 8)}


def test_zero_force_keeps_response_empty_even_when_network_is_nonzero():
    model = FakeEncoder()
    encoder = FrozenNeoForceEncoder(model, [2.] * 6)
    out = encoder(torch.zeros(1, 1, 3, 4, 4), torch.zeros(1, 1, 6, 4, 4), torch.ones(1, 1, 6, 4, 4))
    assert out['tokens'].any()
    assert not out['response'].any()
    assert not model.weight.requires_grad


def test_opposite_sensor_forces_do_not_cancel_contact():
    model = FakeEncoder()
    encoder = FrozenNeoForceEncoder(model, [2.] * 6)
    force = torch.zeros(1, 1, 6, 4, 4)
    force[:, :, 0] = 2
    force[:, :, 3] = -2
    out = encoder(torch.zeros(1, 1, 3, 4, 4), force, torch.ones_like(force))
    assert (out['response'] == 2).all()
    assert model.received['force'][0, 0, 0, 0, 0] == 1
    assert model.received['force'][0, 0, 3, 0, 0] == -1


def test_no_contact_mask_blocks_force_response():
    encoder = FrozenNeoForceEncoder(FakeEncoder(), [1.] * 6)
    out = encoder(torch.zeros(1, 1, 3, 4, 4), torch.ones(1, 1, 6, 4, 4), torch.zeros(1, 1, 6, 4, 4))
    assert not out['response'].any()


def test_reject_unknown_training_scale():
    with pytest.raises(ValueError, match='scale'):
        FrozenNeoForceEncoder(FakeEncoder(), [0.] * 6)


def test_contact_pooling_keeps_sensor_and_time_order_without_cross_sensor_leakage():
    tokens = torch.ones(1, 2, 4, 4, 3)
    tokens[:, 1] *= 7
    response = torch.zeros(1, 2, 2, 8, 8)
    response[0, 0, 0, 0, 0] = 1
    response[0, 1, 1, 7, 7] = 2
    out = pool_contact_tokens({'tokens': tokens, 'response': response}, (2, 2))
    assert out['neoforce'].shape == (1, 2, 2, 2, 2, 3)
    assert out['neoforce'][0, 0, 0, 0, 0].tolist() == [1., 1., 1.]
    assert out['neoforce'][0, 1, 1, 1, 1].tolist() == [7., 7., 7.]
    assert out['neoforce'].count_nonzero() == 6
    assert out['response'][0, 1, 1, 1, 1] == 2


def test_contact_pooling_cannot_turn_empty_response_into_contact():
    out = pool_contact_tokens({'tokens': torch.ones(1, 1, 2, 2, 4),
                               'response': torch.zeros(1, 1, 2, 4, 4)}, (1, 1))
    assert not out['neoforce'].any()


def test_causal_contact_endpoints_never_read_future_or_duplicate_context():
    calls = []
    def encoder(rgb, force, mask):
        values = rgb[0, :, 0, 0, 0]
        calls.append(values.tolist())
        return {'tokens': values.reshape(1, -1, 1, 1, 1),
                'response': torch.ones(1, len(values), 2, 1, 1)}
    rgb = torch.arange(8.).reshape(1, 8, 1, 1, 1).expand(-1, -1, 3, 1, 1)
    force = torch.ones(1, 8, 6, 1, 1)
    out = encode_causal_contact_frames(encoder, rgb, force, force, [0, 3, 7], (1, 1))
    assert calls == [[0.], [0., 1., 2., 3.], [4., 5., 6., 7.]]
    assert out['neoforce'].shape == (1, 2, 3, 1, 1, 1)
    assert out['neoforce'][0, 0, :, 0, 0, 0].tolist() == [0., 3., 7.]
    assert out['raw_frame_ids'].tolist() == [0, 3, 7]


@pytest.mark.parametrize('anchors', [[-1], [8], [3, 3], [4, 2], [1.5], []])
def test_bad_contact_anchors_fail_before_encoder(anchors):
    def encoder(*args):
        pytest.fail('encoder must not run for invalid anchors')
    rgb = torch.zeros(1, 8, 3, 1, 1)
    force = torch.zeros(1, 8, 6, 1, 1)
    with pytest.raises(ValueError, match='anchors'):
        encode_causal_contact_frames(encoder, rgb, force, force, anchors, (1, 1))
