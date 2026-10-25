from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DISEASE_KNOWLEDGE_PATH, LOCAL_EXPERT_CONTACTS_PATH

GUIDANCE_FIELDS = (
    "symptoms",
    "look_alikes",
    "immediate_steps",
    "management",
    "monitoring",
    "prevention",
    "safety_precautions",
)
OPTIONAL_GUIDANCE_LIST_FIELDS = (
    "cultural_practices",
    "biological_options",
    "chemical_options",
    "follow_up_inspection",
    "what_not_to_do",
)


def load_disease_knowledge(path: str | Path = DISEASE_KNOWLEDGE_PATH) -> dict[str, dict[str, Any]]:
    knowledge_path = Path(path)
    if not knowledge_path.exists():
        return {}
    with knowledge_path.open("r", encoding="utf-8") as knowledge_file:
        document = json.load(knowledge_file)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported disease knowledge format: {knowledge_path}")
    records = document.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"Disease knowledge 'records' must be an object: {knowledge_path}")

    for class_name, record in records.items():
        if not isinstance(class_name, str) or not class_name or not isinstance(record, dict):
            raise ValueError(f"Invalid disease knowledge record in {knowledge_path}")
        if record.get("verification_status") != "verified":
            continue
        required = (
            *GUIDANCE_FIELDS,
            "crop",
            "disease",
            "disease_category",
            "distinguishing_signs",
            "favorable_conditions",
            "source_name",
            "source_url",
            "last_reviewed",
        )
        if any(not isinstance(record.get(field), str) or not record[field].strip() for field in required[:2]):
            raise ValueError(f"Verified guidance for {class_name!r} requires symptom and look-alike details.")
        for field in GUIDANCE_FIELDS[2:]:
            values = record.get(field)
            if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
                raise ValueError(f"Verified guidance field {field!r} for {class_name!r} must be a list of text.")
        for field in OPTIONAL_GUIDANCE_LIST_FIELDS:
            if field in record and (
                not isinstance(record[field], list)
                or not all(isinstance(item, str) and item.strip() for item in record[field])
            ):
                raise ValueError(f"Optional verified guidance field {field!r} for {class_name!r} must be a list.")
        for field in (
            "crop",
            "disease",
            "disease_category",
            "distinguishing_signs",
            "favorable_conditions",
            "source_name",
            "source_url",
            "last_reviewed",
        ):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Verified guidance for {class_name!r} is missing {field!r}.")
        if not record["source_url"].startswith("https://"):
            raise ValueError(f"Verified guidance for {class_name!r} must use an HTTPS source URL.")
    return records


def get_verified_guidance(
    class_name: str,
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    knowledge_records = load_disease_knowledge() if records is None else records
    record = knowledge_records.get(class_name)
    if record is None or record.get("verification_status") != "verified":
        return None
    return record


def is_healthy_class(class_name: str) -> bool:
    _crop, separator, condition = class_name.partition("___")
    return bool(separator and condition.strip().lower() == "healthy")


def get_verified_weather_rule(
    class_name: str,
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    knowledge_records = load_disease_knowledge() if records is None else records
    record = get_verified_guidance(class_name, knowledge_records)
    if record is None:
        return None
    rule = record.get("weather_risk_rule")
    if not isinstance(rule, dict):
        return None
    required = ("summary", "source_name", "source_url", "last_reviewed")
    if any(not isinstance(rule.get(field), str) or not rule[field].strip() for field in required):
        return None
    if not rule["source_url"].startswith("https://"):
        return None
    return rule


def load_local_expert_contacts(
    path: str | Path = LOCAL_EXPERT_CONTACTS_PATH,
) -> list[dict[str, str]]:
    contacts_path = Path(path)
    if not contacts_path.exists():
        return []
    with contacts_path.open("r", encoding="utf-8") as contacts_file:
        document = json.load(contacts_file)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported local expert contacts format: {contacts_path}")
    contacts = document.get("contacts")
    if not isinstance(contacts, list):
        raise ValueError(f"Local expert contacts must be a list: {contacts_path}")

    validated_contacts = []
    for contact in contacts:
        if not isinstance(contact, dict):
            raise ValueError(f"Invalid local expert contact in {contacts_path}")
        name = contact.get("name")
        organization = contact.get("organization")
        url = contact.get("url")
        if not all(isinstance(value, str) and value.strip() for value in (name, organization, url)):
            raise ValueError(f"Each local expert contact needs a name, organization, and URL: {contacts_path}")
        if not url.startswith("https://"):
            raise ValueError(f"Local expert contact URLs must use HTTPS: {contacts_path}")
        phone = contact.get("phone")
        if phone is not None and (not isinstance(phone, str) or not phone.strip()):
            raise ValueError(f"Local expert contact phone values must be non-empty text: {contacts_path}")
        validated_contacts.append(
            {
                "name": name.strip(),
                "organization": organization.strip(),
                "url": url.strip(),
                **({"phone": phone.strip()} if phone is not None else {}),
            }
        )
    return validated_contacts
