import json

import pytest

from src.cropshield.knowledge_base import load_local_expert_contacts


def test_contacts_file_can_be_empty_without_fabricating_contacts(tmp_path):
    path = tmp_path / "contacts.json"
    path.write_text(json.dumps({"schema_version": 1, "contacts": []}), encoding="utf-8")

    assert load_local_expert_contacts(path) == []


def test_contacts_accept_only_named_https_sources(tmp_path):
    path = tmp_path / "contacts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contacts": [
                    {
                        "name": "Extension office",
                        "organization": "Verified local university",
                        "url": "https://example.edu/extension",
                        "phone": "+1 555 0100",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_local_expert_contacts(path)[0]["name"] == "Extension office"


def test_contacts_reject_http_urls(tmp_path):
    path = tmp_path / "contacts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contacts": [{"name": "Office", "organization": "University", "url": "http://example.edu"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        load_local_expert_contacts(path)
