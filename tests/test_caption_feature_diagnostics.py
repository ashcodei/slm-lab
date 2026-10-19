import torch
from slm_lab.caption_feature_diagnostics import geometry


def test_geometry_distinguishes_collapsed_from_diverse_caption_features():
    collapsed = geometry(torch.ones(8, 16))
    diverse = geometry(torch.eye(8))
    assert abs(collapsed['mean_off_diagonal_cosine'] - 1) < 1e-9
    assert collapsed['centered_effective_rank'] == 0
    assert diverse['mean_off_diagonal_cosine'] == 0
    assert abs(diverse['centered_effective_rank'] - 7) < 1e-8
    assert diverse['normalized_centered_total_variance'] > 0.8
