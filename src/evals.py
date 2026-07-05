"""Lightweight evaluation case loading and response scoring."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """A single Slack workflow evaluation case."""

    id: str
    message: str
    expected_intent: str | None = None
    expected_agent: str | None = None
    expected_confirmation_required: bool | None = None
    expected_background: bool | None = None
    expected_model_profile: str | None = None
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        """Build and validate an evaluation case."""
        case_id = str(data.get("id", "")).strip()
        message = str(data.get("message", "")).strip()
        if not case_id:
            raise ValueError("Eval case is missing required field 'id'")
        if not message:
            raise ValueError(f"Eval case {case_id!r} is missing required field 'message'")

        return cls(
            id=case_id,
            message=message,
            expected_intent=_optional_string(data.get("expected_intent")),
            expected_agent=_optional_string(data.get("expected_agent")),
            expected_confirmation_required=_optional_bool(data.get("expected_confirmation_required")),
            expected_background=_optional_bool(data.get("expected_background")),
            expected_model_profile=_optional_string(data.get("expected_model_profile")),
            must_include=_string_list(data.get("must_include", []), "must_include", case_id),
            must_not_include=_string_list(data.get("must_not_include", []), "must_not_include", case_id),
            notes=str(data.get("notes", "")).strip(),
        )


@dataclass(frozen=True)
class EvalPrediction:
    """Saved model/bot output for one eval case."""

    id: str
    response: str = ""
    intent: str | None = None
    agent: str | None = None
    confirmation_required: bool | None = None
    background: bool | None = None
    model_profile: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path, line_number: int) -> "EvalPrediction":
        """Build and validate a saved eval prediction."""
        case_id = str(data.get("id", "")).strip()
        if not case_id:
            raise ValueError(f"Prediction on {path}:{line_number} is missing field 'id'")
        return cls(
            id=case_id,
            response=str(data.get("response", "")).strip(),
            intent=_optional_string(data.get("intent")),
            agent=_optional_string(data.get("agent")),
            confirmation_required=_optional_bool(data.get("confirmation_required")),
            background=_optional_bool(data.get("background")),
            model_profile=_optional_string(data.get("model_profile")),
        )


@dataclass(frozen=True)
class EvalScore:
    """Text-response score for one eval case."""

    case_id: str
    passed: bool
    missing_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _string_list(value: Any, field_name: str, case_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Eval case {case_id!r} field {field_name!r} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL eval cases from a file."""
    cases: list[EvalCase] = []
    path = Path(path)

    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {e}") from e
            cases.append(EvalCase.from_dict(data))

    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate eval case ids in {path}: {', '.join(duplicates)}")

    return cases


def load_response_map(path: str | Path) -> dict[str, str]:
    """Load JSONL responses keyed by eval case id."""
    predictions = load_eval_predictions(path)
    return {
        case_id: prediction.response
        for case_id, prediction in predictions.items()
    }


def load_eval_predictions(path: str | Path) -> dict[str, EvalPrediction]:
    """Load JSONL predictions keyed by eval case id."""
    predictions: dict[str, EvalPrediction] = {}
    path = Path(path)

    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {e}") from e

            prediction = EvalPrediction.from_dict(data, path, line_number)
            predictions[prediction.id] = prediction

    return predictions


def score_text_response(case: EvalCase, response: str) -> EvalScore:
    """Score one response using simple inclusion/exclusion checks."""
    normalized = response.lower()
    missing = [
        term
        for term in case.must_include
        if term.lower() not in normalized
    ]
    forbidden = [
        term
        for term in case.must_not_include
        if term.lower() in normalized
    ]
    return EvalScore(
        case_id=case.id,
        passed=not missing and not forbidden,
        missing_terms=missing,
        forbidden_terms=forbidden,
    )


def score_responses(cases: list[EvalCase], responses: dict[str, str]) -> list[EvalScore]:
    """Score response text for all cases that have response entries."""
    return [
        score_text_response(case, responses.get(case.id, ""))
        for case in cases
    ]


def score_prediction(case: EvalCase, prediction: EvalPrediction | None) -> EvalScore:
    """Score one saved prediction against text and routing expectations."""
    prediction = prediction or EvalPrediction(id=case.id)
    text_score = score_text_response(case, prediction.response)
    mismatches: list[str] = []

    _compare_expected("intent", case.expected_intent, prediction.intent, mismatches)
    _compare_expected("agent", case.expected_agent, prediction.agent, mismatches)
    _compare_expected("model_profile", case.expected_model_profile, prediction.model_profile, mismatches)
    _compare_expected(
        "confirmation_required",
        case.expected_confirmation_required,
        prediction.confirmation_required,
        mismatches,
    )
    _compare_expected("background", case.expected_background, prediction.background, mismatches)

    return EvalScore(
        case_id=case.id,
        passed=text_score.passed and not mismatches,
        missing_terms=text_score.missing_terms,
        forbidden_terms=text_score.forbidden_terms,
        mismatches=mismatches,
    )


def score_predictions(
    cases: list[EvalCase],
    predictions: dict[str, EvalPrediction],
) -> list[EvalScore]:
    """Score saved predictions for all cases."""
    return [
        score_prediction(case, predictions.get(case.id))
        for case in cases
    ]


def _compare_expected(
    field_name: str,
    expected: Any,
    actual: Any,
    mismatches: list[str],
) -> None:
    if expected is None:
        return
    if actual != expected:
        mismatches.append(f"{field_name}: expected {expected!r}, got {actual!r}")
