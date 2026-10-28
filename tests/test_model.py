import numpy as np
import pytest
import torch

from digits_cnn.model import DigitCNN, predict, split_indices


def test_network_has_finite_logits_and_gradients():
    torch.manual_seed(42)
    model = DigitCNN()
    logits = model(torch.rand(4, 1, 8, 8))
    assert logits.shape == (4, 10)
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_partitions_are_disjoint_and_cover_input():
    labels = np.tile(np.arange(10), 30)
    fit, valid, test = split_indices(labels)
    assert set(fit).isdisjoint(valid) and set(fit).isdisjoint(test) and set(valid).isdisjoint(test)
    assert len(set(fit) | set(valid) | set(test)) == len(labels)


def test_safe_checkpoint_roundtrip_and_probability_contract(tmp_path):
    (tmp_path / "artifacts").mkdir()
    torch.save(DigitCNN().state_dict(), tmp_path / "artifacts/cnn.pt")
    result = predict(tmp_path, {"pixels": np.zeros((8, 8)).tolist()})
    assert 0 <= result["digit"] < 10
    assert sum(result["probabilities"]) == pytest.approx(1.0)


@pytest.mark.parametrize("pixels", [np.zeros((7, 8)), np.full((8, 8), 17), np.full((8, 8), np.nan)])
def test_invalid_images_rejected_before_loading_model(tmp_path, pixels):
    with pytest.raises(ValueError):
        predict(tmp_path, {"pixels": pixels.tolist()})
