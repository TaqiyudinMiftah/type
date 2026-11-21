"""Protect the no-target-state-update contract of frozen linear probes."""

import numpy as np
import pytest
import torch

from ketik_audio.common import ResearchError
from ketik_audio.frozen import (
    assert_unchanged,
    extract,
    fit_head,
    frozen_encoder,
    state_snapshot,
)
from ketik_audio.stability import variant_model


def toy_model():
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
        ),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )


def test_extraction_and_head_fit_never_update_backbone_or_batchnorm():
    model = toy_model()
    before = state_snapshot(model)
    encoder = frozen_encoder(model)
    images = np.random.default_rng(44).normal(size=(48, 8, 8)).astype(np.float32)
    features = extract(
        encoder, images, np.zeros((1, 8, 1), np.float32), np.ones((1, 8, 1), np.float32), "cpu"
    )
    fit_head(features[:36], np.tile(np.arange(3), 12), features[36:])
    assert_unchanged(model, before)
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    assert model[0][1].num_batches_tracked.item() == 0
    assert features.shape == (48, 4)


def test_removing_classifier_preserves_the_original_model_factorization():
    model = toy_model().eval()
    x = torch.randn(5, 1, 8, 8)
    with torch.inference_mode():
        expected = model(x)
        encoder = frozen_encoder(model)
        actual = model[-1](encoder(x))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_accidentally_reenabled_training_is_rejected_before_target_forward():
    encoder = frozen_encoder(toy_model())
    before = state_snapshot(encoder)
    encoder.train()
    with pytest.raises(ResearchError, match="mode train"):
        extract(encoder, np.ones((2, 8, 8), np.float32), 0, 1, "cpu")
    assert_unchanged(encoder, before)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_activation_control_preserves_every_initial_weight_and_changes_one_module(seed):
    original, _ = variant_model(seed, "relu")
    modified, _ = variant_model(seed, "leaky")
    assert_unchanged(modified, state_snapshot(original))
    left, right = dict(original.named_modules()), dict(modified.named_modules())
    changed = [name for name in left if type(left[name]) is not type(right[name])]
    assert changed == ["0.mlp4.1"]
    assert right["0.mlp4.1"].negative_slope == 0.01
