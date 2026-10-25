from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.cropshield.dataset import (
    collect_image_records,
    detect_dataset_root,
    extract_label_from_path,
    extract_plantvillage_color_archive,
    inspect_dataset_integrity,
    summarize_dataset,
)
from src.cropshield.train import build_datasets, stratified_split
from scripts.train_model import select_training_classes


def _write_dataset(root: Path, images_per_class: int = 8) -> None:
    for class_index, class_name in enumerate(("Crop___healthy", "Crop___rust")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(images_per_class):
            pixels = np.full((24, 24, 3), class_index * 80 + image_index, dtype=np.uint8)
            Image.fromarray(pixels).save(class_dir / f"{image_index}.png")


def test_extract_label_from_path():
    label = extract_label_from_path("data/PlantVillage/Apple___Apple_scab/0001.jpg")
    assert label == "Apple___Apple_scab"


def test_summarize_dataset_missing(tmp_path):
    result = summarize_dataset(tmp_path / "does-not-exist")
    assert result["status"] == "missing"
    assert result["class_count"] == 0


def test_explicit_unsupported_dataset_is_rejected_by_prepare_command(tmp_path, capsys):
    from scripts.prepare_dataset import main

    invalid_root = tmp_path / "empty"
    invalid_root.mkdir()
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["prepare_dataset.py", "--dataset-root", str(invalid_root)]
        assert main() == 1
    finally:
        sys.argv = original_argv
    assert "No supported dataset layout" in capsys.readouterr().out


def test_detect_dataset_root_supports_repository_raw_color_layout(tmp_path):
    dataset_root = tmp_path / "PlantVillage-Dataset" / "raw" / "color"
    dataset_root.mkdir(parents=True)
    _write_dataset(dataset_root, images_per_class=3)
    assert detect_dataset_root(tmp_path / "PlantVillage-Dataset") == dataset_root.resolve()


def test_extract_plantvillage_archive_extracts_only_color_class_images(tmp_path):
    archive_path = tmp_path / "plantvillage.zip"
    image_bytes = b"color-image"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "plantvillage dataset/color/Crop___healthy/leaf.jpg",
            image_bytes,
        )
        archive.writestr(
            "plantvillage dataset/grayscale/Crop___healthy/leaf.jpg",
            b"grayscale-image",
        )
        archive.writestr(
            "plantvillage dataset/color/../outside.jpg",
            b"unsafe-path",
        )

    destination, count = extract_plantvillage_color_archive(
        archive_path,
        tmp_path / "extracted",
        progress_interval=1,
    )

    assert count == 1
    assert (destination / "Crop___healthy" / "leaf.jpg").read_bytes() == image_bytes
    assert not (tmp_path / "outside.jpg").exists()
    assert not (destination / "grayscale").exists()


def test_extract_plantvillage_archive_requires_color_images(tmp_path):
    archive_path = tmp_path / "grayscale-only.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("plantvillage dataset/grayscale/Crop___healthy/leaf.jpg", b"image")

    with pytest.raises(ValueError, match="No color JPG"):
        extract_plantvillage_color_archive(archive_path, tmp_path / "extracted")


def test_exact_duplicates_are_excluded_before_split(tmp_path):
    _write_dataset(tmp_path)
    class_dir = tmp_path / "Crop___healthy"
    (class_dir / "duplicate.png").write_bytes((class_dir / "0.png").read_bytes())
    records = collect_image_records(tmp_path)
    assert len(records) == 16
    splits = stratified_split(records, seed=7)
    paths_by_split = {name: {record["path"] for record in values} for name, values in splits.items()}
    assert not (paths_by_split["train"] & paths_by_split["validation"])
    assert not (paths_by_split["train"] & paths_by_split["test"])
    assert not (paths_by_split["validation"] & paths_by_split["test"])


def test_integrity_audit_reports_dimensions_and_duplicates(tmp_path):
    _write_dataset(tmp_path, images_per_class=3)
    class_dir = tmp_path / "Crop___healthy"
    (class_dir / "copy.png").write_bytes((class_dir / "0.png").read_bytes())
    result = inspect_dataset_integrity(tmp_path)
    assert result["valid_images"] == 7
    assert len(result["exact_duplicates"]) == 1
    assert result["dimensions"] == {"24x24": 7}


def test_dataset_preparation_writes_labels_metadata_and_per_image_quality_report(tmp_path):
    from scripts.prepare_dataset import write_dataset_reports

    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_dataset(dataset_root, images_per_class=3)

    metadata = write_dataset_reports(dataset_root, output_dir)

    assert metadata["class_count"] == 2
    assert metadata["valid_images"] == 6
    assert (output_dir / "class_labels.json").exists()
    assert (output_dir / "dataset_metadata.json").exists()
    assert (output_dir / "split_manifest.csv").exists()
    assert metadata["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    report = (output_dir / "image_quality_report.csv").read_text(encoding="utf-8")
    assert "brightness,blur_score,glare_fraction" in report
    assert "Crop___healthy" in report


def test_knowledge_sync_marks_missing_classes_unavailable_without_inventing_guidance(tmp_path):
    import json

    from scripts.prepare_dataset import synchronize_unverified_knowledge_records

    path = tmp_path / "disease_knowledge.json"
    path.write_text(
        json.dumps({"schema_version": 1, "region": "India", "records": {}}),
        encoding="utf-8",
    )
    added = synchronize_unverified_knowledge_records(["Tomato___Early_blight"], path)
    records = json.loads(path.read_text(encoding="utf-8"))["records"]

    assert added == 1
    assert records["Tomato___Early_blight"]["verification_status"] == "unavailable"
    assert records["Tomato___Early_blight"]["management"] == []
    assert records["Tomato___Early_blight"]["source_url"] == ""


def test_build_datasets_creates_stratified_split(tmp_path):
    _write_dataset(tmp_path)
    train, validation, test, labels, mapping, splits = build_datasets(
        tmp_path,
        image_size=(32, 32),
        batch_size=2,
        seed=42,
    )
    assert labels == ["Crop___healthy", "Crop___rust"]
    assert mapping[labels[0]] == 0
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 10,
        "validation": 2,
        "test": 4,
    }
    batch_images, batch_labels = next(iter(train))
    assert tuple(batch_images.shape) == (2, 32, 32, 3)
    assert batch_labels.shape == (2,)
    assert sum(len(items) for items in (train, validation, test)) > 0


def test_build_datasets_can_limit_training_to_selected_classes(tmp_path):
    _write_dataset(tmp_path)
    extra_class = tmp_path / "Crop___other"
    extra_class.mkdir()
    for index in range(8):
        Image.fromarray(np.full((24, 24, 3), 200 + index, dtype=np.uint8)).save(
            extra_class / f"{index}.png"
        )

    train, validation, test, labels, mapping, splits = build_datasets(
        tmp_path,
        image_size=(32, 32),
        batch_size=2,
        class_names=["Crop___rust", "Crop___healthy"],
    )

    assert labels == ["Crop___healthy", "Crop___rust"]
    assert mapping == {"Crop___healthy": 0, "Crop___rust": 1}
    assert all(
        record["label"] in {"Crop___healthy", "Crop___rust"}
        for split in splits.values()
        for record in split
    )


def test_collect_image_records_rejects_unknown_selected_class(tmp_path):
    _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="Selected dataset class folders were not found"):
        collect_image_records(tmp_path, class_names=["Crop___missing"])


def test_training_class_selection_selects_only_the_requested_crop(tmp_path):
    _write_dataset(tmp_path)
    tomato = tmp_path / "Tomato___healthy"
    tomato.mkdir()
    for index in range(3):
        Image.fromarray(np.full((24, 24, 3), 220 + index, dtype=np.uint8)).save(
            tomato / f"{index}.png"
        )

    assert select_training_classes(tmp_path, "Crop") == ["Crop___healthy", "Crop___rust"]
    assert select_training_classes(tmp_path, "all") is None
