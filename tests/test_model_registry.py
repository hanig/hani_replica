"""Tests for configured model profiles."""

import pytest


def test_model_registry_has_expected_profiles():
    """Runtime model registry exposes named profiles for routing."""
    from src.config import MODEL_REGISTRY

    expected = {"router", "agent", "briefing", "deep_research", "ideaspark", "heavy"}
    assert expected.issubset(MODEL_REGISTRY)


def test_legacy_model_constants_match_registry():
    """Backwards-compatible constants resolve through the registry."""
    from src.config import (
        AGENT_MODEL,
        BRIEFING_MODEL,
        DEEP_RESEARCH_MODEL,
        HEAVY_AGENT_MODEL,
        IDEASPARK_MODEL,
        INTENT_MODEL,
        get_model_id,
    )

    assert INTENT_MODEL == get_model_id("router")
    assert AGENT_MODEL == get_model_id("agent")
    assert BRIEFING_MODEL == get_model_id("briefing")
    assert DEEP_RESEARCH_MODEL == get_model_id("deep_research")
    assert IDEASPARK_MODEL == get_model_id("ideaspark")
    assert HEAVY_AGENT_MODEL == get_model_id("heavy")


def test_unknown_model_profile_raises_clear_error():
    """Unknown profiles should fail with an actionable message."""
    from src.config import get_model_profile

    with pytest.raises(ValueError, match="Unknown model profile"):
        get_model_profile("missing")


def test_model_profiles_are_serializable():
    """Profiles can be exposed by get_config without leaking secrets."""
    from src.config import get_config

    config = get_config()
    router = config["model_registry"]["router"]
    assert set(router) == {"name", "provider", "model", "max_tokens", "description"}
    assert router["provider"] == "anthropic"
