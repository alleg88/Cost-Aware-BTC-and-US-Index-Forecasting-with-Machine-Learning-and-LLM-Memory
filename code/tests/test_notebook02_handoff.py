import json

import pandas as pd
import pytest


def _width_rows():
    return pd.DataFrame(
        {
            "width_bps": [75, 65, 55],
            "sortino": [-1.97, -2.29, -4.49],
            "sharpe": [-1.45, -1.64, -3.25],
            "net_return": [-0.077, -0.115, -0.321],
            "trades": [267, 480, 878],
        }
    )


def test_notebook01_width_handoff_is_exact_and_round_trips(tmp_path):
    from experiments.notebook02_handoff import (
        load_notebook01_handoff,
        write_notebook01_handoff,
    )

    path = tmp_path / "selected_widths.parquet"
    written = write_notebook01_handoff(_width_rows(), path)
    loaded = load_notebook01_handoff(path)

    assert written == path
    assert loaded["width_bps"].tolist() == [75, 65, 55]
    pd.testing.assert_frame_equal(loaded, _width_rows())


@pytest.mark.parametrize("bad_widths", ([75, 65], [75, 65, 60], [55, 65, 75, 85]))
def test_notebook01_width_handoff_rejects_any_other_width_set(tmp_path, bad_widths):
    from experiments.notebook02_handoff import write_notebook01_handoff

    frame = pd.DataFrame(
        {
            "width_bps": bad_widths,
            "sortino": [-1.0] * len(bad_widths),
            "sharpe": [-1.0] * len(bad_widths),
            "net_return": [-0.1] * len(bad_widths),
            "trades": [50] * len(bad_widths),
        }
    )
    with pytest.raises(ValueError, match="DZ55, DZ65 and DZ75"):
        write_notebook01_handoff(frame, tmp_path / "bad.parquet")


def test_direct_pipeline_handoff_freezes_180_days_and_candidate_pool(tmp_path):
    from experiments.catboost_matched_ablation import CANDIDATE_POOL_FINGERPRINT
    from experiments.notebook02_handoff import (
        BASE_FEATURE_COLUMNS,
        load_pipeline_handoff,
        write_pipeline_handoff,
    )

    path = tmp_path / "handoff.json"
    write_pipeline_handoff(
        path=path,
        upstream=_width_rows(),
    )
    payload = load_pipeline_handoff(path)
    assert payload["sentiment"] == "none"
    assert payload["training_histories_days"] == {"55": 180, "65": 180, "75": 180}
    assert payload["candidate_pool_fingerprint"] == CANDIDATE_POOL_FINGERPRINT
    assert tuple(payload["features"]["columns"]) == BASE_FEATURE_COLUMNS

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["candidate_pool_fingerprint"] = "changed"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate pool fingerprint"):
        load_pipeline_handoff(path)
