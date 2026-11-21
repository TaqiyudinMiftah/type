"""Regressions for input signal loss and source/target selection boundaries."""

import numpy as np
import pytest
import torch

from ketik_audio.cnn_repair import repaired_model as compact_model
from ketik_audio.repair import development_split, input_probe, qualifies, repaired_model
from ketik_audio.stem_repair import repaired_model as wide_stem_model


@pytest.mark.parametrize(
    ("builder", "variant"),
    [
        (repaired_model, "xavier-leaky"),
        (repaired_model, "residual-prenorm"),
        (compact_model, "compact-resnet"),
    ],
)
def test_repair_preserves_input_signal_and_gradient(builder, variant):
    torch.set_num_threads(2)
    rng = np.random.default_rng(123)
    mel = rng.normal(size=(36, 64, 64)).astype(np.float32)
    y = np.arange(36)
    mean, std = np.zeros((1, 64, 1), np.float32), np.ones((1, 64, 1), np.float32)
    original, _ = repaired_model(42, "original")
    original_probe = input_probe(original, mel, y, mean, std, "cpu")
    model, _ = builder(42, variant)
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    probe = input_probe(model, mel, y, mean, std, "cpu")
    assert probe["max_logit_range"] > 1e-5
    assert probe["input_gradient_l2"] > 1000 * original_probe["input_gradient_l2"]
    assert probe["stem_gradient_l2"] > 0
    assert all(torch.equal(tensor, buffers[name]) for name, tensor in model.named_buffers())


def test_xavier_patch_only_changes_attention_projection_weights():
    reference, _ = repaired_model(43, "leaky")
    repaired, _ = repaired_model(43, "xavier-leaky")
    before, after = reference.state_dict(), repaired.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    expected = {
        f"0.s{stage}.fc_{projection}.weight"
        for stage in (3, 4)
        for projection in ("q", "k", "v", "o")
    }
    assert changed == expected
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in repaired.parameters()
    )


def test_development_support_is_disjoint_from_validation_and_unused_test():
    y, order = np.repeat(np.arange(36), 25), np.tile(np.arange(25), 36)
    split = development_split(y, order)
    assert np.array_equal(np.bincount(y[split["support"]]), np.full(36, 10))
    assert len(split["validation"]) == 72
    assert len(split["unused_test"]) == 180
    assert not set(split["support"]) & set(split["validation"])
    assert not set(split["support"]) & set(split["unused_test"])


def test_selection_rejects_a_collapsed_or_missing_initialization():
    rows = [
        {
            "training_seed": seed,
            "support_accuracy": 0.99,
            "support_max_logit_range": 10,
            "initial_probe": {"input_gradient_l2": 0.1},
            "validation": {"accuracy": 0.75},
        }
        for seed in (42, 43, 44)
    ]
    assert qualifies(rows)
    assert not qualifies(rows[:2])
    rows[1]["support_accuracy"] = 1 / 36
    assert not qualifies(rows)


def test_wide_stem_learns_through_multiple_spatial_filters():
    torch.set_num_threads(2)
    model, _ = wide_stem_model(43, "wide-stem-residual")
    model.eval()
    x = torch.randn(4, 1, 64, 64)
    assert model[0].s0(x).shape == (4, 64, 64, 64)
    assert isinstance(model[0].mlp0, torch.nn.Identity)
    logits = model(x)
    assert logits.shape == (4, 36)
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    grad = model[0].s0[0].weight.grad
    assert torch.isfinite(grad).all()
    assert torch.all(grad.reshape(32, -1).abs().sum(1) > 0)
