"""Tests for evaluation case loading and scoring."""

import json

import pytest

from src.evals import (
    EvalCase,
    EvalPrediction,
    load_eval_cases,
    load_eval_predictions,
    load_response_map,
    score_prediction,
    score_text_response,
)


def test_load_eval_cases(tmp_path):
    """JSONL eval cases are parsed into typed cases."""
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({
            "id": "calendar.next",
            "message": "what is next?",
            "expected_agent": "calendar",
            "expected_background": False,
            "must_include": ["next"],
        })
        + "\n"
    )

    cases = load_eval_cases(path)
    assert len(cases) == 1
    assert cases[0].id == "calendar.next"
    assert cases[0].expected_agent == "calendar"
    assert cases[0].expected_background is False
    assert cases[0].must_include == ["next"]


def test_load_eval_cases_rejects_duplicate_ids(tmp_path):
    """Duplicate ids make eval output ambiguous and should fail."""
    path = tmp_path / "cases.jsonl"
    record = {"id": "dup", "message": "hello"}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="Duplicate eval case ids"):
        load_eval_cases(path)


def test_load_response_map(tmp_path):
    """Saved responses are keyed by eval id."""
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({
            "id": "case-1",
            "response": "hello",
            "agent": "calendar",
            "confirmation_required": False,
        })
        + "\n"
    )

    assert load_response_map(path) == {"case-1": "hello"}
    predictions = load_eval_predictions(path)
    assert predictions["case-1"].agent == "calendar"
    assert predictions["case-1"].confirmation_required is False


def test_score_text_response():
    """Simple inclusion/exclusion checks catch obvious regressions."""
    case = EvalCase(
        id="email.draft",
        message="draft email",
        must_include=["draft"],
        must_not_include=["sent"],
    )

    passed = score_text_response(case, "I created a draft for review.")
    failed = score_text_response(case, "I sent the message.")

    assert passed.passed
    assert not failed.passed
    assert failed.missing_terms == ["draft"]
    assert failed.forbidden_terms == ["sent"]


def test_score_prediction_checks_route_and_safety_metadata():
    """Prediction scoring catches routing and confirmation regressions."""
    case = EvalCase(
        id="github.issue",
        message="create issue",
        expected_agent="github",
        expected_confirmation_required=True,
        expected_background=False,
        expected_model_profile="agent",
        must_include=["confirm"],
    )
    passed = score_prediction(
        case,
        EvalPrediction(
            id="github.issue",
            response="Please confirm creating the issue.",
            agent="github",
            confirmation_required=True,
            background=False,
            model_profile="agent",
        ),
    )
    failed = score_prediction(
        case,
        EvalPrediction(
            id="github.issue",
            response="Created the issue.",
            agent="research",
            confirmation_required=False,
            background=False,
            model_profile="heavy",
        ),
    )

    assert passed.passed
    assert not failed.passed
    assert any("agent" in mismatch for mismatch in failed.mismatches)
    assert any("confirmation_required" in mismatch for mismatch in failed.mismatches)
