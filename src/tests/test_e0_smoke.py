import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from e0_smoke import run


def test_synthetic_smoke_is_deterministic_and_emits_step_records():
    config = {
        "seed": 5,
        "candidate_count": 8,
        "max_steps": 4,
        "mode": "full_risk",
        "alpha": 0.1,
        "lambda_pose": 1.0,
        "lambda_cost": 0.1,
        "lambda_cov": 0.5,
        "initial_quality": 0.35,
        "target_quality": 0.70,
        "max_risk": 0.20,
    }
    first = run(config)
    second = run(config)

    assert first["config_hash"] == second["config_hash"]
    assert [item["selected_view"] for item in first["steps"]] == [item["selected_view"] for item in second["steps"]]
    assert first["steps"]
    assert {"selected_view", "quality_lower_bound", "stop"}.issubset(first["steps"][0])
