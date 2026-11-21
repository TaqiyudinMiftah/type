"""Protect paired comparisons and reject cherry-picked/partial pilot summaries."""

import pytest

from ketik_audio.common import ResearchError
from ketik_audio.resnet_analysis import analyse
from ketik_audio.resnet_fewshot import BLOCKS, KS


def record():
    rows = []
    for block, (seed, draw) in enumerate(BLOCKS, start=1):
        rows.append(
            {
                "block": block,
                "method": "source_only",
                "k": 0,
                "training_seed": seed,
                "support_seed": None,
                "test": {"accuracy": 0.05},
            }
        )
        for k in KS:
            for method in ("scratch", "adapt"):
                # Sign alternates by block; analysis must preserve these pairings.
                accuracy = min(0.5 + k * 0.04, 0.99)
                if method == "adapt":
                    accuracy += {1: 0.06, 2: -0.03, 3: 0.03}[block]
                rows.append(
                    {
                        "block": block,
                        "method": method,
                        "k": k,
                        "training_seed": seed,
                        "support_seed": draw,
                        "support_accuracy": 1.0,
                        "training_seconds": 5,
                        "test": {"accuracy": accuracy, "macro_f1": accuracy, "top5": 1.0},
                    }
                )
    return {
        "status": "complete",
        "conditions": rows,
        "sources": [{"training_seconds": 10} for _ in range(3)],
    }


def test_paired_gain_retains_negative_block_and_threshold_scope():
    summary = analyse(record())
    for row in summary["paired"]:
        assert row["per_block_difference"] == pytest.approx([0.06, -0.03, 0.03])
        assert row["difference"]["mean"] == pytest.approx(0.02)
        assert row["negative_blocks"] == 1
    assert summary["observed_grid_thresholds"]["scratch"]["90"]["first_mean_k"] == 10
    assert summary["observed_grid_thresholds"]["adapt"]["90"]["first_all_blocks_k"] is None
    assert summary["not_independent_test_event_count"] == 360


@pytest.mark.parametrize("change", ["missing", "duplicate", "running"])
def test_analysis_rejects_incomplete_or_duplicate_results(change):
    data = record()
    if change == "missing":
        data["conditions"].pop()
    elif change == "duplicate":
        data["conditions"][-1] = dict(data["conditions"][-2])
    else:
        data["status"] = "running"
    with pytest.raises(ResearchError):
        analyse(data)
