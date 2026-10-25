from datetime import date
import sqlite3

from PIL import Image
import pytest

from src.cropshield import history


def test_history_stores_only_metadata_and_can_be_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    history.append_prediction(
        "Crop___rust",
        "baseline-seed42",
        0.82,
        top_predictions=[{"class_name": "Crop___rust", "probability": 0.82}],
        trust_level="Medium",
        inference_ms=24.5,
        image_quality={"brightness": 120.0},
        user_notes="Check again.",
        image_name="leaf.png",
    )
    rows = history.fetch_history()
    assert len(rows) == 1
    assert rows[0]["predicted_class"] == "Crop___rust"
    assert rows[0]["model_version"] == "baseline-seed42"
    assert rows[0]["confidence"] == pytest.approx(0.82)
    assert rows[0]["top_predictions"][0]["class_name"] == "Crop___rust"
    assert rows[0]["trust_level"] == "Medium"
    assert rows[0]["inference_ms"] == pytest.approx(24.5)
    assert rows[0]["image_quality"] == {"brightness": 120.0}
    assert rows[0]["user_notes"] == "Check again."
    assert rows[0]["image_name"] == "leaf.png"
    assert "image" not in rows[0]
    history.clear_history()
    assert history.fetch_history() == []


def test_history_rejects_nonpositive_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    with pytest.raises(ValueError, match="positive"):
        history.fetch_history(limit=0)


def test_plant_timeline_stores_notes_and_only_explicitly_retained_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(history, "PLANT_TIMELINE_IMAGES_DIR", tmp_path / "timeline-images")
    plant_id = history.create_tracked_plant("Tomato bed A")

    without_photo = history.add_plant_observation(plant_id, date(2026, 10, 25), "Two leaves look yellow.")
    with_photo = history.add_plant_observation(
        plant_id,
        date(2026, 10, 26),
        "New photo for comparison.",
        predicted_class="Tomato___Early_blight",
        confidence=0.62,
        retained_image=Image.new("RGB", (24, 20), "green"),
        record_type="ai_prediction",
        trust_level="Medium",
    )

    plants = history.fetch_tracked_plants()
    observations = history.fetch_plant_observations(plant_id)
    assert plants[0]["name"] == "Tomato bed A"
    assert plants[0]["observation_count"] == 2
    assert [item["id"] for item in observations] == [without_photo, with_photo]
    assert observations[0]["image_filename"] is None
    assert observations[0]["predicted_class"] is None
    assert observations[1]["predicted_class"] == "Tomato___Early_blight"
    assert observations[1]["confidence"] == pytest.approx(0.62)
    assert observations[1]["record_type"] == "ai_prediction"
    assert observations[1]["trust_level"] == "Medium"
    saved_image = history.get_plant_timeline_image_path(observations[1]["image_filename"])
    assert saved_image.exists()
    with sqlite3.connect(history.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM plant_observations").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0


def test_deleting_a_timeline_cascades_observations_and_removes_retained_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(history, "PLANT_TIMELINE_IMAGES_DIR", tmp_path / "timeline-images")
    plant_id = history.create_tracked_plant("Pepper plot")
    history.add_plant_observation(
        plant_id,
        date(2026, 10, 25),
        "Photo retained with permission.",
        retained_image=Image.new("RGB", (24, 20), "green"),
    )
    observation = history.fetch_plant_observations(plant_id)[0]
    image_path = history.get_plant_timeline_image_path(observation["image_filename"])

    assert history.delete_tracked_plant(plant_id) == 1
    assert not image_path.exists()
    assert history.fetch_plant_observations(plant_id) == []
    assert history.fetch_tracked_plants() == []


def test_deleting_one_observation_removes_only_its_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(history, "PLANT_TIMELINE_IMAGES_DIR", tmp_path / "timeline-images")
    plant_id = history.create_tracked_plant("Tomato bed")
    first_id = history.add_plant_observation(
        plant_id,
        date(2026, 10, 25),
        "First visit.",
        retained_image=Image.new("RGB", (24, 20), "green"),
    )
    second_id = history.add_plant_observation(plant_id, date(2026, 10, 26), "Second visit.")
    first_image = history.get_plant_timeline_image_path(
        history.fetch_plant_observations(plant_id)[0]["image_filename"]
    )

    assert history.delete_plant_observation(plant_id, first_id) == 1
    assert not first_image.exists()
    assert [row["id"] for row in history.fetch_plant_observations(plant_id)] == [second_id]
    assert history.delete_plant_observation(plant_id, first_id) == 0


def test_timeline_validates_user_input_and_image_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    with pytest.raises(ValueError, match="Plant name"):
        history.create_tracked_plant("  ")
    plant_id = history.create_tracked_plant("Apple tree")
    with pytest.raises(ValueError, match="notes"):
        history.add_plant_observation(plant_id, date.today(), " ")
    with pytest.raises(ValueError, match="between 0 and 1"):
        history.add_plant_observation(plant_id, date.today(), "Check leaves.", confidence=1.1)
    with pytest.raises(ValueError, match="Invalid timeline image"):
        history.get_plant_timeline_image_path("..\\outside.jpg")


def test_plot_map_and_spread_use_ai_predictions_without_promoting_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    tomato = history.create_tracked_plant("Tomato 1", "North field", 1, 2)
    pepper = history.create_tracked_plant("Tomato 2", "North field", 2, 2)
    other_plot = history.create_tracked_plant("Tomato 3", "South field", 1, 1)
    history.add_plant_observation(
        tomato, date(2026, 10, 25), "Model scan.", "Tomato___Early_blight", 0.71,
        record_type="ai_prediction",
    )
    history.add_plant_observation(
        pepper, date(2026, 10, 25), "Model scan.", "Tomato___Early_blight", 0.66,
        record_type="ai_prediction",
    )
    history.add_plant_observation(
        other_plot, date(2026, 10, 25), "Different plot.", "Tomato___Early_blight", 0.68,
        record_type="ai_prediction",
    )
    history.add_plant_observation(
        tomato, date(2026, 10, 26), "User note.", "Tomato___Early_blight", 0.99,
        record_type="user_observation",
    )
    history.add_plant_observation(
        pepper, date(2026, 10, 26), "User observation.", record_type="user_observation",
        user_marked_affected=True,
    )
    history.add_plant_observation(
        pepper, date(2026, 10, 27), "Human expert confirmed the condition.", "Tomato___Early_blight",
        record_type="expert_confirmed",
    )
    history.add_plant_observation(
        pepper, date(2026, 10, 26), "Healthy model scan.", "Tomato___healthy", 0.9,
        record_type="ai_prediction",
    )

    spread = history.fetch_possible_plot_spread("North field")
    map_rows = history.fetch_plot_map("North field")

    assert spread == [
        {"predicted_class": "Tomato___Early_blight", "scan_count": 2, "plant_count": 2}
    ]
    assert history.fetch_possible_plot_spread("South field") == []
    assert len(map_rows) == 2
    assert map_rows[0]["grid_row"] == 1
    assert map_rows[0]["grid_column"] == 2
    assert map_rows[0]["latest_ai_prediction"] == "Tomato___Early_blight"
    assert map_rows[1]["user_marked_affected"] == 1
    assert map_rows[1]["latest_expert_confirmation"] == "Tomato___Early_blight"


def test_action_checklist_persists_completion_and_verified_source(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    plant_id = history.create_tracked_plant("Apple tree")
    checklist_id = history.create_action_checklist(
        plant_id,
        "Apple___example",
        "Example agricultural university",
        "https://example.edu/disease",
        [("Monitoring", "Inspect leaves weekly.")],
    )
    items = history.fetch_action_checklists(plant_id, "Apple___example")
    assert items[0]["checklist_id"] == checklist_id
    assert items[0]["source_name"] == "Example agricultural university"
    history.set_action_checklist_item(items[0]["item_id"], True)
    assert history.fetch_action_checklists(plant_id, "Apple___example")[0]["completed"] == 1


def test_user_affected_mark_cannot_be_attached_to_an_ai_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    plant_id = history.create_tracked_plant("Apple tree")

    with pytest.raises(ValueError, match="Only a user observation"):
        history.add_plant_observation(
            plant_id,
            date.today(),
            "AI scan.",
            "Apple___example",
            record_type="ai_prediction",
            user_marked_affected=True,
        )
