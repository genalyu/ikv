import torch
from n0_twam.preprocessing.rgb_contact_matching import mutual_contact_matches


def test_unique_reciprocal_matches_with_contact_only():
    visual = torch.eye(3)
    tactile = torch.eye(3)[[2, 0, 1]]
    result = mutual_contact_matches(visual, tactile, torch.tensor([1., 1., 0.]))
    assert result['visual_rows'].tolist() == [2, 0, -1]


def test_ambiguous_visual_or_tactile_matches_stay_unmatched():
    assert mutual_contact_matches(torch.ones(3, 2), torch.ones(1, 2), torch.ones(1))['visual_rows'].tolist() == [-1]
    assert mutual_contact_matches(torch.eye(3), torch.tensor([[1., 0., 0.]]).repeat(2, 1), torch.ones(2))['visual_rows'].tolist() == [-1, -1]


def test_all_zero_visual_is_not_a_match():
    result = mutual_contact_matches(torch.zeros(3, 2), torch.ones(2, 2), torch.ones(2))
    assert result['visual_rows'].tolist() == [-1, -1]
    assert torch.isfinite(result['scores']).all() and torch.isfinite(result['margins']).all()
