import json

import pytest

from src.cropshield.knowledge_base import (
    get_verified_guidance,
    get_verified_weather_rule,
    is_healthy_class,
    load_disease_knowledge,
)


def _record():
    return {
        "verification_status": "verified",
        "crop": "Tomato",
        "disease": "Example disease",
        "disease_category": "Fungal",
        "symptoms": "Example symptoms.",
        "distinguishing_signs": "Example diagnostic signs.",
        "favorable_conditions": "Example favorable conditions.",
        "look_alikes": "Example look-alike.",
        "immediate_steps": ["Ask an expert to confirm the diagnosis."],
        "management": ["Use only reviewed management guidance."],
        "monitoring": ["Check whether symptoms spread."],
        "prevention": ["Use source-backed preventive measures."],
        "safety_precautions": ["Follow local expert advice."],
        "source_name": "Example agricultural university",
        "source_url": "https://example.edu/disease",
        "last_reviewed": "2026-10-24",
    }


def test_load_and_lookup_verified_disease_guidance(tmp_path):
    path = tmp_path / "disease_knowledge.json"
    path.write_text(
        json.dumps({"schema_version": 1, "region": "India", "records": {"Tomato___example": _record()}}),
        encoding="utf-8",
    )

    records = load_disease_knowledge(path)

    assert get_verified_guidance("Tomato___example", records)["source_name"] == "Example agricultural university"
    assert get_verified_guidance("Unknown___class", records) is None


def test_unverified_record_is_not_returned_as_guidance():
    records = {"Tomato___example": {"verification_status": "unreviewed"}}

    assert get_verified_guidance("Tomato___example", records) is None


def test_healthy_class_is_not_treated_as_a_disease():
    assert is_healthy_class("Tomato___healthy")
    assert not is_healthy_class("Tomato___Early_blight")
    assert not is_healthy_class("healthy")


def test_verified_guidance_requires_https_source(tmp_path):
    path = tmp_path / "disease_knowledge.json"
    record = _record()
    record["source_url"] = "http://example.edu/disease"
    path.write_text(
        json.dumps({"schema_version": 1, "region": "India", "records": {"Tomato___example": record}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        load_disease_knowledge(path)


def test_weather_rule_requires_a_verified_disease_record_and_citable_source():
    rule = {
        "summary": "A source-backed weather condition.",
        "source_name": "Example agricultural university",
        "source_url": "https://example.edu/weather",
        "last_reviewed": "2026-10-24",
    }
    record = _record()
    record["weather_risk_rule"] = rule

    assert get_verified_weather_rule("Tomato___example", {"Tomato___example": record}) == rule
    assert get_verified_weather_rule(
        "Tomato___example",
        {"Tomato___example": {**record, "verification_status": "unreviewed"}},
    ) is None
    assert get_verified_weather_rule(
        "Tomato___example",
        {"Tomato___example": {**record, "weather_risk_rule": {**rule, "source_url": "http://example.edu"}}},
    ) is None


def test_optional_verified_guidance_lists_are_validated(tmp_path):
    path = tmp_path / "disease_knowledge.json"
    record = _record()
    record["chemical_options"] = ["Only if confirmed and locally approved."]
    path.write_text(
        json.dumps({"schema_version": 1, "region": "India", "records": {"Tomato___example": record}}),
        encoding="utf-8",
    )

    assert load_disease_knowledge(path)["Tomato___example"]["chemical_options"]

    record["chemical_options"] = "not a reviewed list"
    path.write_text(
        json.dumps({"schema_version": 1, "region": "India", "records": {"Tomato___example": record}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Optional verified guidance"):
        load_disease_knowledge(path)
