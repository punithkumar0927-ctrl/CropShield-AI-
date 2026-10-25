from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_home_app_smoke():
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project_root / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("CropShield AI" in title.value for title in app.title)


def test_plant_timeline_page_smoke():
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project_root / "pages" / "8_Plant_Timeline.py")).run(timeout=30)
    assert not app.exception
    assert any("Plant Follow-up Timeline" in title.value for title in app.title)


def test_detection_page_shows_one_actionable_missing_model_message():
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project_root / "pages" / "1_Disease_Detection.py")).run(timeout=45)

    assert not app.exception
    rendered_messages = [element.value for element in (*app.warning, *app.info, *app.error)]
    assert not any("preprocessing_config.json is missing" in message for message in rendered_messages)
    model_path = project_root / "models" / "crop_model.keras"
    if model_path.is_file():
        assert not any("No trained CropShield model is available" in message for message in rendered_messages)
    else:
        assert any("No trained CropShield model is available" in message for message in rendered_messages)
        assert any("How to train the model" in expander.label for expander in app.expander)
