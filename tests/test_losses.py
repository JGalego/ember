import pytest
import torch

from ember.models.losses import (
    build_loss,
    contrastive_loss,
    hinge_loss,
    info_nce_loss,
    margin_ranking_loss,
)

LOSS_FNS = {
    "contrastive": contrastive_loss,
    "margin_ranking": margin_ranking_loss,
    "info_nce": info_nce_loss,
    "hinge": hinge_loss,
}


@pytest.mark.parametrize("name,fn", LOSS_FNS.items())
def test_loss_prefers_well_separated_energies(name, fn):
    # A batch where positives already have much lower energy than negatives
    # should incur less loss than the reverse (negatives lower than positives).
    e_pos_good = torch.zeros(8)
    e_neg_good = torch.full((8,), 5.0)
    e_pos_bad = torch.full((8,), 5.0)
    e_neg_bad = torch.zeros(8)

    loss_good = fn(e_pos_good, e_neg_good)
    loss_bad = fn(e_pos_bad, e_neg_bad)
    assert loss_good.dim() == 0
    assert loss_good.item() < loss_bad.item()


@pytest.mark.parametrize("name,fn", LOSS_FNS.items())
def test_loss_supports_multiple_negatives_per_positive(name, fn):
    e_pos = torch.zeros(4)
    e_neg = torch.rand(4, 3) + 2.0  # (B, K) negatives
    loss = fn(e_pos, e_neg)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("name", list(LOSS_FNS))
def test_build_loss_registry(name):
    fn = build_loss(name)
    assert fn is LOSS_FNS[name]


def test_build_loss_rejects_unknown_kind():
    with pytest.raises(KeyError):
        build_loss("not_a_real_loss")
