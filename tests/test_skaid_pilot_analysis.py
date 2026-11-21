import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_pilot_analysis import analyse
from ketik_audio.skaid_pilot_common import DRAWS, ENCODERS, KS, SEEDS


def synthetic_trials():
    trials = []
    for encoder in ENCODERS:
        for subject in range(17):
            for permuted, ks in ((False, KS), (True, (10,))):
                for k in ks:
                    for draw in DRAWS:
                        value = subject / 20
                        trials.append(
                            {
                                "encoder": encoder,
                                "subject": subject,
                                "session_token": f"s{subject}",
                                "k": k,
                                "draw": draw,
                                "permuted": permuted,
                                "support_fit_label_accuracy": value,
                                "support_true_label_accuracy": value,
                                "query": {
                                    "examples": 60,
                                    "accuracy": value,
                                    "macro_f1": value,
                                    "top3": value,
                                },
                            }
                        )
    zero = [
        {"source_seed": seed, "subject": subject, "query": {"accuracy": subject / 20}}
        for seed in SEEDS
        for subject in range(17)
    ]
    return trials, zero


def test_id_aggregation_keeps_encoder_and_draw_repeats_inside_id():
    trials, zero = synthetic_trials()
    result = analyse(trials, zero)
    for group in result["groups"]:
        assert group["accuracy"]["mean"] == pytest.approx(0.4)
        assert group["accuracy"]["ids"] == 17
        assert group["accuracy"]["sd_ids"] == pytest.approx(np.std(np.arange(17) / 20, ddof=1))
    for row in result["per_id"]:
        assert row["predictors_averaged"] == (9 if row["family"] in ("pretrained", "random") else 3)
    assert all(
        x["first_minus_second"]["mean"] == pytest.approx(0) for x in result["paired_comparisons"]
    )
    assert all(v["0.8"] is None for v in result["thresholds_on_fixed_grid"].values())


def test_missing_or_duplicate_predictor_is_rejected():
    trials, zero = synthetic_trials()
    with pytest.raises(ResearchError):
        analyse(trials[:-1], zero)
    with pytest.raises(ResearchError):
        analyse(trials + [trials[0]], zero)
