from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropshield.config import DATA_DIR, DEFAULT_SEED  # noqa: E402
from src.cropshield.dataset import (  # noqa: E402
    detect_dataset_root,
    extract_plantvillage_color_archive,
    inspect_dataset_integrity,
    iter_class_dirs,
    stratified_split,
    summarize_dataset,
    validate_image_file,
)
from src.cropshield.image_quality import assess_image_quality  # noqa: E402


def write_split_manifest(
    records: list[dict[str, object]],
    output_dir: Path = DATA_DIR,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    splits = stratified_split(records, seed=seed)
    split_path = output_dir / "split_manifest.csv"
    columns = ("path", "label", "width", "height", "split")
    with split_path.open("w", encoding="utf-8", newline="") as split_file:
        writer = csv.DictWriter(split_file, fieldnames=columns)
        writer.writeheader()
        for split_name, split_items in splits.items():
            for record in split_items:
                writer.writerow({**record, "split": split_name})
    return {name: len(values) for name, values in splits.items()}


def write_split_manifest_from_quality_report(
    report_path: Path,
    output_dir: Path = DATA_DIR,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    records = []
    with report_path.open("r", encoding="utf-8", newline="") as report_file:
        for row in csv.DictReader(report_file):
            if row["valid"].lower() != "true" or row["exact_duplicate"].lower() == "true":
                continue
            records.append(
                {
                    "path": row["image_path"],
                    "label": row["class_label"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )
    counts = write_split_manifest(records, output_dir, seed)
    metadata_path = output_dir / "dataset_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split_seed"] = seed
        metadata["split_counts"] = counts
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return counts


def synchronize_unverified_knowledge_records(
    labels: list[str],
    knowledge_path: Path = DATA_DIR / "disease_knowledge.json",
) -> int:
    """Add unavailable placeholders for model labels without inventing disease advice."""
    if knowledge_path.exists():
        document = json.loads(knowledge_path.read_text(encoding="utf-8"))
    else:
        document = {"schema_version": 1, "region": "India", "records": {}}
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported disease knowledge format: {knowledge_path}")
    records = document.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"Disease knowledge records must be an object: {knowledge_path}")

    unavailable = "Not source-verified; consult a qualified local agricultural expert."
    added = 0
    for label in labels:
        if label in records:
            continue
        crop, separator, condition = label.partition("___")
        records[label] = {
            "crop": crop if separator else "Not identified",
            "disease": condition.replace("_", " ") if separator else label,
            "disease_category": "unavailable",
            "symptoms": unavailable,
            "distinguishing_signs": unavailable,
            "favorable_conditions": unavailable,
            "look_alikes": unavailable,
            "immediate_steps": [],
            "management": [],
            "prevention": [],
            "monitoring": [],
            "safety_precautions": [],
            "when_to_contact_expert": unavailable,
            "source_name": "",
            "source_url": "",
            "last_reviewed": "",
            "verification_status": "unavailable",
        }
        added += 1
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return added


def write_dataset_reports(dataset_root: Path, output_dir: Path = DATA_DIR) -> dict[str, object]:
    classes = iter_class_dirs(dataset_root)
    if not classes:
        raise ValueError("No class folders containing supported images were found.")
    audit = inspect_dataset_integrity(dataset_root)
    labels = [class_dir.name for class_dir in classes]
    class_counts = {
        class_dir.name: sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for class_dir in classes
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    knowledge_path = output_dir / "disease_knowledge.json"
    unavailable_records_added = synchronize_unverified_knowledge_records(labels, knowledge_path)

    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_count": len(labels),
        "class_labels": labels,
        "class_counts": class_counts,
        "total_images": audit["total_images"],
        "valid_images": audit["valid_images"],
        "invalid_image_count": len(audit["invalid_images"]),
        "exact_duplicate_count": len(audit["exact_duplicates"]),
        "knowledge_records_added_as_unavailable": unavailable_records_added,
        "image_dimensions": audit["dimensions"],
        "quality_metrics_note": (
            "Quality metrics are simple capture heuristics, not a plant detector or disease-specific suitability test."
        ),
    }
    invalid_paths = set(audit["invalid_images"])
    duplicate_paths = set(audit["exact_duplicates"])
    split_records: list[dict[str, object]] = []
    report_path = output_dir / "image_quality_report.csv"
    columns = (
        "image_path",
        "class_label",
        "valid",
        "validation_issue",
        "exact_duplicate",
        "width",
        "height",
        "brightness",
        "blur_score",
        "glare_fraction",
        "green_pixel_fraction",
        "quality_acceptable",
        "feedback",
    )
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=columns)
        writer.writeheader()
        processed = 0
        for class_dir in classes:
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for image_path in image_paths:
                valid, issue, width, height = validate_image_file(image_path)
                row = {
                    "image_path": str(image_path.resolve()),
                    "class_label": class_dir.name,
                    "valid": valid,
                    "validation_issue": "" if valid else issue,
                    "exact_duplicate": str(image_path) in duplicate_paths,
                    "width": width,
                    "height": height,
                    "brightness": "",
                    "blur_score": "",
                    "glare_fraction": "",
                    "green_pixel_fraction": "",
                    "quality_acceptable": "",
                    "feedback": "",
                }
                if valid and str(image_path) not in invalid_paths:
                    try:
                        with Image.open(image_path) as image:
                            quality = assess_image_quality(image)
                        row.update(
                            {
                                "brightness": quality.brightness,
                                "blur_score": quality.blur_score,
                                "glare_fraction": quality.glare_fraction,
                                "green_pixel_fraction": quality.green_pixel_fraction,
                                "quality_acceptable": quality.acceptable,
                                "feedback": "; ".join(quality.feedback),
                            }
                        )
                    except (OSError, UnidentifiedImageError):
                        row["valid"] = False
                        row["validation_issue"] = "Image could not be reopened for quality analysis."
                writer.writerow(row)
                if row["valid"] and not row["exact_duplicate"]:
                    split_records.append(
                        {
                            "path": str(image_path.resolve()),
                            "label": class_dir.name,
                            "width": width,
                            "height": height,
                        }
                    )
                processed += 1
                if processed % 5000 == 0:
                    print(f"Audited image quality for {processed:,} files.")
    metadata["split_seed"] = DEFAULT_SEED
    metadata["split_counts"] = write_split_manifest(split_records, output_dir)
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or validate the PlantVillage dataset.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset-root",
        type=str,
        help="Path to an extracted PlantVillage dataset directory.",
    )
    source.add_argument(
        "--dataset-archive",
        type=str,
        help="Path to a PlantVillage ZIP archive; only the original color images are extracted.",
    )
    args = parser.parse_args()
    if args.dataset_archive:
        destination = DATA_DIR / "PlantVillage-Dataset" / "color"
        root, image_count = extract_plantvillage_color_archive(args.dataset_archive, destination)
        print(f"Extracted {image_count:,} color images to: {root}")
    else:
        root = (
            detect_dataset_root(Path(args.dataset_root).expanduser())
            if args.dataset_root
            else detect_dataset_root()
        )
    if root is None:
        print(
            "No supported dataset layout was found. Point --dataset-root at an extracted dataset "
            "or --dataset-archive at a PlantVillage ZIP archive."
        )
        return 1
    summary = summarize_dataset(root)
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Class count: {summary['class_count']}")
    print(f"Image count: {sum(summary['class_counts'].values()):,}")
    print(f"Class counts: {summary['class_counts']}")
    metadata = write_dataset_reports(Path(str(summary["dataset_root"])))
    print(f"Valid images: {metadata['valid_images']:,}")
    print(f"Unreadable images: {metadata['invalid_image_count']:,}")
    print(f"Exact duplicates: {metadata['exact_duplicate_count']:,}")
    print(
        "Knowledge records added as unavailable (no treatment claims invented): "
        f"{metadata['knowledge_records_added_as_unavailable']}"
    )
    print(f"Dataset metadata and quality report written under: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
