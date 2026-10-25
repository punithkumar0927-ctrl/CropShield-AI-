from __future__ import annotations

import io
import json
import logging
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from PIL import Image

from .config import PLANT_TIMELINE_IMAGES_DIR, ROOT_DIR

DB_PATH = ROOT_DIR / "crop_history.db"
LOGGER = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            model_version TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    prediction_columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    for column_name, definition in (
        ("top_predictions", "TEXT"),
        ("trust_level", "TEXT"),
        ("inference_ms", "REAL"),
        ("image_quality", "TEXT"),
        ("user_notes", "TEXT"),
        ("image_name", "TEXT"),
    ):
        if column_name not in prediction_columns:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {column_name} {definition}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracked_plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plant_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER NOT NULL REFERENCES tracked_plants(id) ON DELETE CASCADE,
            observed_on TEXT NOT NULL,
            notes TEXT NOT NULL,
            predicted_class TEXT,
            confidence REAL,
            record_type TEXT NOT NULL DEFAULT 'user_observation',
            trust_level TEXT,
            user_marked_affected INTEGER,
            image_filename TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    plant_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tracked_plants)")}
    for column_name, definition in (
        ("plot_name", "TEXT NOT NULL DEFAULT ''"),
        ("grid_row", "INTEGER"),
        ("grid_column", "INTEGER"),
    ):
        if column_name not in plant_columns:
            conn.execute(f"ALTER TABLE tracked_plants ADD COLUMN {column_name} {definition}")
    observation_columns = {row["name"] for row in conn.execute("PRAGMA table_info(plant_observations)")}
    for column_name, definition in (
        ("record_type", "TEXT NOT NULL DEFAULT 'user_observation'"),
        ("trust_level", "TEXT"),
        ("user_marked_affected", "INTEGER"),
    ):
        if column_name not in observation_columns:
            conn.execute(f"ALTER TABLE plant_observations ADD COLUMN {column_name} {definition}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS action_checklists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER NOT NULL REFERENCES tracked_plants(id) ON DELETE CASCADE,
            class_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS action_checklist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checklist_id INTEGER NOT NULL REFERENCES action_checklists(id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            action TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            source_name TEXT NOT NULL,
            source_url TEXT NOT NULL
        )
        """
    )
    return conn


def append_prediction(
    predicted_class: str,
    model_version: str,
    confidence: float,
    top_predictions: list[dict[str, Any]] | None = None,
    trust_level: str | None = None,
    inference_ms: float | None = None,
    image_quality: dict[str, Any] | None = None,
    user_notes: str = "",
    image_name: str | None = None,
) -> None:
    if not predicted_class.strip() or not model_version.strip():
        raise ValueError("Prediction class and model version are required.")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if inference_ms is not None and inference_ms < 0:
        raise ValueError("Inference time cannot be negative.")
    if trust_level is not None and trust_level not in {"High", "Medium", "Low", "Cannot Assess"}:
        raise ValueError("Trust level is invalid.")
    if len(user_notes) > 2000:
        raise ValueError("Prediction notes must be at most 2,000 characters.")
    normalized_image_name = Path(image_name).name[:255] if image_name else None
    serialized_predictions = json.dumps(top_predictions) if top_predictions is not None else None
    serialized_quality = json.dumps(image_quality) if image_quality is not None else None
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (timestamp, predicted_class, model_version, confidence, top_predictions,
                    trust_level, inference_ms, image_quality, user_notes, image_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, predicted_class, model_version, float(confidence),
                    serialized_predictions, trust_level, inference_ms, serialized_quality,
                    user_notes.strip(), normalized_image_name,
                ),
            )


def fetch_history(limit: int = 20) -> List[Dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be a positive integer.")
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT timestamp, predicted_class, model_version, confidence, top_predictions,
                   trust_level, inference_ms, image_quality, user_notes, image_name
            FROM predictions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        row["top_predictions"] = json.loads(row["top_predictions"]) if row["top_predictions"] else []
        row["image_quality"] = json.loads(row["image_quality"]) if row["image_quality"] else {}
    return result


def clear_history() -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute("DELETE FROM predictions")


def create_tracked_plant(
    name: str,
    plot_name: str | None = None,
    grid_row: int | None = None,
    grid_column: int | None = None,
) -> int:
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 80:
        raise ValueError("Plant name must contain 1 to 80 characters.")
    normalized_plot = (plot_name or normalized_name).strip()
    if not normalized_plot or len(normalized_plot) > 80:
        raise ValueError("Plot name must contain 1 to 80 characters.")
    if (grid_row is None) != (grid_column is None):
        raise ValueError("Both field-map row and column must be supplied together.")
    if grid_row is not None and (not 1 <= grid_row <= 100 or not 1 <= grid_column <= 100):
        raise ValueError("Field-map row and column must be between 1 and 100.")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO tracked_plants (name, plot_name, grid_row, grid_column, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_name, normalized_plot, grid_row, grid_column, timestamp),
            )
            return int(cursor.lastrowid)


def fetch_tracked_plants() -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.plot_name, p.grid_row, p.grid_column,
                   p.created_at, COUNT(o.id) AS observation_count
            FROM tracked_plants AS p
            LEFT JOIN plant_observations AS o ON o.plant_id = p.id
            GROUP BY p.id
            ORDER BY p.name COLLATE NOCASE, p.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_plot_map(plot_name: str) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.grid_row, p.grid_column,
                   (SELECT o.user_marked_affected
                    FROM plant_observations o
                    WHERE o.plant_id = p.id AND o.record_type = 'user_observation'
                      AND o.user_marked_affected IS NOT NULL
                    ORDER BY o.observed_on DESC, o.id DESC LIMIT 1) AS user_marked_affected,
                   (SELECT o.predicted_class
                    FROM plant_observations o
                    WHERE o.plant_id = p.id AND o.record_type = 'ai_prediction'
                      AND o.predicted_class IS NOT NULL
                    ORDER BY o.observed_on DESC, o.id DESC LIMIT 1) AS latest_ai_prediction,
                   (SELECT o.predicted_class
                     FROM plant_observations o
                     WHERE o.plant_id = p.id AND o.record_type = 'expert_confirmed'
                       AND o.predicted_class IS NOT NULL
                     ORDER BY o.observed_on DESC, o.id DESC LIMIT 1) AS latest_expert_confirmation
            FROM tracked_plants p
            WHERE p.plot_name = ?
            ORDER BY p.grid_row, p.grid_column, p.name
            """,
            (plot_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_possible_plot_spread(plot_name: str) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT predicted_class, COUNT(*) AS scan_count,
                   COUNT(DISTINCT plant_id) AS plant_count
            FROM plant_observations
            WHERE plant_id IN (SELECT id FROM tracked_plants WHERE plot_name = ?)
              AND record_type = 'ai_prediction'
              AND predicted_class IS NOT NULL
              AND lower(substr(predicted_class, instr(predicted_class, '___') + 3)) != 'healthy'
            GROUP BY predicted_class
            HAVING COUNT(*) >= 2
            ORDER BY scan_count DESC, predicted_class
            """,
            (plot_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_plant_observation(
    plant_id: int,
    observed_on: date,
    notes: str,
    predicted_class: str | None = None,
    confidence: float | None = None,
    retained_image: Image.Image | None = None,
    record_type: str = "user_observation",
    user_marked_affected: bool | None = None,
    trust_level: str | None = None,
) -> int:
    normalized_notes = notes.strip()
    if not normalized_notes or len(normalized_notes) > 2000:
        raise ValueError("Observation notes must contain 1 to 2,000 characters.")
    if predicted_class is not None and len(predicted_class) > 200:
        raise ValueError("Predicted class must be at most 200 characters.")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if record_type not in {"ai_prediction", "user_observation", "expert_confirmed"}:
        raise ValueError("Observation type must be AI prediction, user observation, or expert confirmed.")
    if trust_level is not None and trust_level not in {"High", "Medium", "Low", "Cannot Assess"}:
        raise ValueError("Trust level is invalid.")
    if trust_level is not None and record_type != "ai_prediction":
        raise ValueError("Trust level can only be recorded with an AI prediction.")
    if user_marked_affected is not None and record_type != "user_observation":
        raise ValueError("Only a user observation can record a user-marked affected status.")
    if not isinstance(observed_on, date):
        raise ValueError("Observation date must be a calendar date.")

    image_filename = None
    image_path = None
    if retained_image is not None:
        image_filename = f"{uuid4().hex}.jpg"
        image_path = PLANT_TIMELINE_IMAGES_DIR / image_filename
        PLANT_TIMELINE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        preview = retained_image.convert("RGB")
        preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        image_buffer = io.BytesIO()
        preview.save(image_buffer, format="JPEG", quality=82, optimize=True)
        image_path.write_bytes(image_buffer.getvalue())

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    committed = False
    try:
        with closing(_connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO plant_observations
                        (plant_id, observed_on, notes, predicted_class, confidence,
                         record_type, trust_level, user_marked_affected, image_filename, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plant_id,
                        observed_on.isoformat(),
                        normalized_notes,
                        predicted_class or None,
                        confidence,
                        record_type,
                        trust_level,
                        None if user_marked_affected is None else int(user_marked_affected),
                        image_filename,
                        timestamp,
                    ),
                )
                observation_id = int(cursor.lastrowid)
        committed = True
        return observation_id
    finally:
        if image_path is not None and not committed:
            image_path.unlink(missing_ok=True)


def fetch_plant_observations(plant_id: int) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT id, plant_id, observed_on, notes, predicted_class, confidence,
                   record_type, trust_level, user_marked_affected, image_filename, created_at
            FROM plant_observations
            WHERE plant_id = ?
            ORDER BY observed_on, id
            """,
            (plant_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_action_checklist(
    plant_id: int,
    class_name: str,
    source_name: str,
    source_url: str,
    actions: list[tuple[str, str]],
) -> int:
    if not actions or any(not category.strip() or not action.strip() for category, action in actions):
        raise ValueError("A checklist requires non-empty actions from verified guidance.")
    if not source_name.strip() or not source_url.startswith("https://"):
        raise ValueError("Checklist actions require a verified HTTPS source.")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO action_checklists (plant_id, class_name, created_at) VALUES (?, ?, ?)",
                (plant_id, class_name, timestamp),
            )
            checklist_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO action_checklist_items
                    (checklist_id, category, action, source_name, source_url)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (checklist_id, category.strip(), action.strip(), source_name.strip(), source_url.strip())
                    for category, action in actions
                ],
            )
    return checklist_id


def fetch_action_checklists(plant_id: int, class_name: str) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT c.id AS checklist_id, c.class_name, c.created_at, i.id AS item_id,
                   i.category, i.action, i.completed, i.source_name, i.source_url
            FROM action_checklists c
            JOIN action_checklist_items i ON i.checklist_id = c.id
            WHERE c.plant_id = ? AND c.class_name = ?
            ORDER BY c.id DESC, i.id
            """,
            (plant_id, class_name),
        ).fetchall()
    return [dict(row) for row in rows]


def set_action_checklist_item(item_id: int, completed: bool) -> None:
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "UPDATE action_checklist_items SET completed = ? WHERE id = ?",
                (int(completed), item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Action checklist item {item_id} was not found.")


def get_plant_timeline_image_path(image_filename: str) -> Path:
    if Path(image_filename).name != image_filename or not image_filename.endswith(".jpg"):
        raise ValueError("Invalid timeline image reference.")
    image_path = PLANT_TIMELINE_IMAGES_DIR / image_filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Saved timeline photo is missing: {image_filename}")
    return image_path


def delete_plant_observation(plant_id: int, observation_id: int) -> int:
    with closing(_connect()) as conn:
        with conn:
            row = conn.execute(
                """
                SELECT image_filename
                FROM plant_observations
                WHERE id = ? AND plant_id = ?
                """,
                (observation_id, plant_id),
            ).fetchone()
            if row is None:
                return 0
            conn.execute(
                "DELETE FROM plant_observations WHERE id = ? AND plant_id = ?",
                (observation_id, plant_id),
            )
    filename = row["image_filename"]
    if filename:
        try:
            get_plant_timeline_image_path(filename).unlink(missing_ok=True)
        except FileNotFoundError:
            LOGGER.warning("Timeline image %s was already missing during deletion.", filename)
    return 1


def delete_tracked_plant(plant_id: int) -> int:
    with closing(_connect()) as conn:
        with conn:
            rows = conn.execute(
                "SELECT image_filename FROM plant_observations WHERE plant_id = ?",
                (plant_id,),
            ).fetchall()
            cursor = conn.execute("DELETE FROM tracked_plants WHERE id = ?", (plant_id,))
            deleted = cursor.rowcount
    if deleted:
        for row in rows:
            filename = row["image_filename"]
            if filename:
                try:
                    get_plant_timeline_image_path(filename).unlink(missing_ok=True)
                except FileNotFoundError:
                    LOGGER.warning("Timeline image %s was already missing during deletion.", filename)
    return deleted
