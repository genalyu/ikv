import pytest
import torch
from n0_twam.preprocessing.kv_index import observed_index, route_contact_index


def route(rows, response):
    index = observed_index({'dino': torch.arange(6.).reshape(3, 2)}, 3, 'cpu')
    return route_contact_index(index, torch.tensor([[2., 4.], [6., 8.]]),
                               response, torch.tensor(rows, dtype=torch.long),
                               torch.tensor([0, 1, 0]), (0,))


def test_third_person_shares_index_wrist_keeps_tactile_only():
    visual, tactile, paired = route([0, 1], [1., 1.])
    assert paired.tolist() == [True, False]
    assert torch.equal(visual['dino'], torch.arange(6.).reshape(3, 2))
    assert visual['neoforce'].tolist() == [[2., 4.], [0., 0.], [0., 0.]]
    assert tactile['neoforce'].tolist() == [[2., 4.], [6., 8.]]
    assert tactile['dino'].tolist() == [[0., 1.], [0., 0.]]


def test_unknown_and_zero_response_are_distinct():
    visual, tactile, paired = route([-1, 0], [1., 0.])
    assert not visual['neoforce'].any()
    assert tactile['neoforce'].tolist() == [[2., 4.], [0., 0.]]
    assert not paired.any()


def test_multiple_contacts_merge_on_same_existing_row():
    visual, tactile, paired = route([2, 2], [1., 3.])
    assert visual['neoforce'][2].tolist() == [5., 7.]
    assert tactile['neoforce'].tolist() == [[2., 4.], [6., 8.]]
    assert tactile['dino'].tolist() == [[4., 5.], [4., 5.]]


def test_invalid_correspondence_is_not_silently_clamped():
    with pytest.raises(ValueError, match='visual_rows'):
        route([3, -1], [1., 1.])
