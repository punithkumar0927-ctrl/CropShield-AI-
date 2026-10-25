from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import DATA_DIR, DEFAULT_SEED, MAX_IMAGE_PIXELS, ROOT_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PLANTVILLAGE_COLOR_PREFIX = "plantvillage dataset/color/"


def stratified_split(
    records: list[dict[str, object]],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, object]]]:
    """Create deterministic 70/15/15 splits by class after duplicate removal."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    labels = sorted({str(record["label"]) for record in records})
    for label in labels:
        class_records = [record for record in records if record["label"] == label]
        if len(class_records) < 3:
            raise ValueError(
                f"Class '{label}' has only {len(class_records)} valid unique images; at least 3 are required for train/validation/test."
            )
        order = rng.permutation(len(class_records))
        shuffled = [class_records[index] for index in order]
        train_count = max(1, math.floor(len(shuffled) * 0.70))
        validation_count = max(1, math.floor(len(shuffled) * 0.15))
        if train_count + validation_count >= len(shuffled):
            train_count = len(shuffled) - 2
            validation_count = 1
        splits["train"].extend(shuffled[:train_count])
        splits["validation"].extend(shuffled[train_count : train_count + validation_count])
        splits["test"].extend(shuffled[train_count + validation_count :])
    return splits


def extract_plantvillage_color_archive(
    archive_path: str | Path,
    destination: str | Path,
    progress_interval: int = 5000,
) -> Tuple[Path, int]:
    """Extract only the original color images from a PlantVillage ZIP archive."""
    archive_file = Path(archive_path).expanduser().resolve()
    output_root = Path(destination).expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_file}")
    if progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer.")

    output_root.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    try:
        with ZipFile(archive_file) as archive:
            members = []
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.startswith(PLANTVILLAGE_COLOR_PREFIX):
                    continue
                relative_path = PurePosixPath(entry.filename[len(PLANTVILLAGE_COLOR_PREFIX) :])
                if (
                    relative_path.is_absolute()
                    or len(relative_path.parts) != 2
                    or any(part in {"", ".", ".."} for part in relative_path.parts)
                    or Path(relative_path.name).suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue
                members.append((entry, relative_path))

            if not members:
                raise ValueError(
                    "No color JPG, JPEG, or PNG images were found under "
                    f"'{PLANTVILLAGE_COLOR_PREFIX}' in {archive_file}."
                )

            for entry, relative_path in members:
                target = output_root.joinpath(*relative_path.parts)
                if not target.resolve(strict=False).is_relative_to(output_root):
                    raise ValueError(f"Unsafe dataset archive path: {entry.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as destination_file:
                    shutil.copyfileobj(source, destination_file)
                extracted_count += 1
                if extracted_count % progress_interval == 0 or extracted_count == len(members):
                    print(f"Extracted {extracted_count:,} of {len(members):,} color images.")
    except BadZipFile as error:
        raise ValueError(f"Dataset archive is not a valid ZIP file: {archive_file}") from error

    return output_root, extracted_count


def detect_dataset_root(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Find a PlantVillage-like dataset directory in a few common locations."""
    if base_dir is not None:
        provided = Path(base_dir)
        search_roots = [
            provided,
            provided / "raw" / "color",
            provided / "color",
            provided / "PlantVillage",
            provided / "PlantVillage-Dataset",
            provided / "plantvillage",
            provided / "dataset",
            provided / "train",
        ]
    else:
        search_roots = [
            DATA_DIR,
            ROOT_DIR,
            ROOT_DIR / "PlantVillage-Dataset",
            ROOT_DIR / "plantvillage",
            ROOT_DIR / "data" / "PlantVillage-Dataset",
            ROOT_DIR / "data" / "plantvillage",
        ]
    expanded_roots: List[Path] = []
    for candidate in search_roots:
        expanded_roots.extend(
            [
                candidate,
                candidate / "raw" / "color",
                candidate / "color",
                candidate / "PlantVillage",
                candidate / "PlantVillage-Dataset",
                candidate / "plantvillage",
                candidate / "dataset",
                candidate / "train",
            ]
        )
    seen = set()
    for candidate in expanded_roots:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            if _looks_like_plantvillage(resolved):
                return resolved
    return None


def _looks_like_plantvillage(root: Path) -> bool:
    if not root.exists():
        return False
    class_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")][:10]
    if not class_dirs:
        return False
    image_count = 0
    for folder in class_dirs:
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_count += 1
                break
    if image_count == 0:
        return False
    # PlantVillage class folders use the "Crop___Disease" naming. Require it whenever any
    # subfolder matches, so unrelated project folders with a few images are never detected
    # as a dataset root.
    if any("___" in folder.name for folder in class_dirs):
        return all("___" in folder.name for folder in class_dirs if _folder_has_images(folder))
    # Fallback for renamed/extracted layouts without the naming convention: require a
    # realistic dataset, not a stray folder with 2-3 images.
    return image_count >= 5


def _folder_has_images(folder: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"} for item in folder.iterdir()
        )
    except OSError:
        return False


def iter_class_dirs(root: Path) -> List[Path]:
    """Return class directories only when the dataset layout is compatible."""
    if not root.exists() or not root.is_dir():
        return []
    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in {"__MACOSX", ".git"}
    ]
    class_dirs = []
    for d in dirs:
        if any(file.suffix.lower() in {".jpg", ".jpeg", ".png"} for file in d.iterdir() if file.is_file()):
            class_dirs.append(d)
    return sorted(class_dirs)


def extract_label_from_path(path: str | Path) -> str:
    return Path(path).parent.name


def summarize_dataset(root: Optional[Path] = None) -> Dict[str, object]:
    dataset_root = detect_dataset_root(root)
    result: Dict[str, object] = {
        "status": "missing",
        "dataset_root": None,
        "class_count": 0,
        "class_counts": {},
        "sample_files": [],
        "invalid_images": None,
        "duplicate_images": None,
        "image_dimensions": {},
    }
    if dataset_root is None:
        return result
    result["status"] = "available"
    result["dataset_root"] = str(dataset_root)
    class_dirs = iter_class_dirs(dataset_root)
    class_counts: Dict[str, int] = {}
    sample_files: List[str] = []
    dimensions: Dict[str, int] = {}
    for class_dir in class_dirs:
        files = sorted(
            file
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        class_counts[class_dir.name] = len(files)
        if len(sample_files) < 6:
            sample_files.extend(str(file.resolve()) for file in files[: 6 - len(sample_files)])
    result["class_count"] = len(class_dirs)
    result["class_counts"] = class_counts
    result["sample_files"] = sample_files[:6]
    result["image_dimensions"] = dimensions
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_records(
    root: Path,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Read valid image metadata and reject exact duplicate image contents."""
    class_dirs = iter_class_dirs(root)
    if class_names is not None:
        if len(class_names) != len(set(class_names)):
            raise ValueError("Selected class names must be unique.")
        available = {directory.name: directory for directory in class_dirs}
        missing = sorted(set(class_names) - set(available))
        if missing:
            raise ValueError(f"Selected dataset class folders were not found: {missing}.")
        class_dirs = [available[name] for name in sorted(class_names)]
    if not class_dirs:
        raise ValueError(
            "Unsupported dataset layout. Expected class folders containing JPG, JPEG, or PNG images."
        )
    records: List[Dict[str, object]] = []
    seen_hashes: Dict[str, str] = {}
    duplicate_paths: List[str] = []
    invalid_paths: List[str] = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            valid, _, width, height = validate_image_file(image_path)
            if not valid:
                invalid_paths.append(str(image_path))
                continue
            digest = _file_sha256(image_path)
            if digest in seen_hashes:
                previous = seen_hashes[digest]
                if extract_label_from_path(previous) != class_dir.name:
                    raise ValueError(
                        f"Identical image content has conflicting labels: {previous} and {image_path}"
                    )
                duplicate_paths.append(str(image_path))
                continue
            seen_hashes[digest] = str(image_path)
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "label": class_dir.name,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )
    if not records:
        raise ValueError(f"No valid JPG, JPEG, or PNG images were found under {root}.")
    if invalid_paths:
        print(f"Skipped {len(invalid_paths)} unreadable images.")
    if duplicate_paths:
        print(f"Excluded {len(duplicate_paths)} exact duplicate images to prevent split leakage.")
    return records


def inspect_dataset_integrity(root: Path) -> Dict[str, object]:
    """Perform a full image decode, dimension, and exact-content duplicate audit."""
    class_dirs = iter_class_dirs(root)
    if not class_dirs:
        raise ValueError("Unsupported dataset layout. Expected class folders containing image files.")
    image_count = 0
    invalid_paths: List[str] = []
    duplicate_paths: List[str] = []
    dimension_counts: Dict[str, int] = {}
    content_hashes: Dict[str, tuple[str, str]] = {}
    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_count += 1
            valid, _, width, height = validate_image_file(path)
            if not valid:
                invalid_paths.append(str(path))
                continue
            dimension = f"{width}x{height}"
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
            digest = _file_sha256(path)
            previous = content_hashes.get(digest)
            if previous is not None:
                if previous[1] != class_dir.name:
                    raise ValueError(
                        f"Identical image content has conflicting labels: {previous[0]} and {path}"
                    )
                duplicate_paths.append(str(path))
            else:
                content_hashes[digest] = (str(path), class_dir.name)
    return {
        "total_images": image_count,
        "valid_images": image_count - len(invalid_paths),
        "invalid_images": invalid_paths,
        "exact_duplicates": duplicate_paths,
        "dimensions": dimension_counts,
    }


def validate_image_file(path: str | Path) -> Tuple[bool, str, int, int]:
    file_path = Path(path)
    if not file_path.exists():
        return False, "File does not exist.", 0, 0
    suffix = file_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        return False, "Unsupported file type. Use JPG, JPEG, or PNG.", 0, 0
    try:
        with Image.open(file_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                return False, "Image dimensions exceed the supported 40-megapixel limit.", 0, 0
            image.verify()
            return True, "OK", width, height
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return False, "Image file is corrupt or unreadable.", 0, 0


def list_dataset_samples(root: Optional[Path] = None, limit: int = 18) -> List[str]:
    dataset_root = detect_dataset_root(root)
    if dataset_root is None:
        return []
    class_dirs = iter_class_dirs(dataset_root)
    samples: List[str] = []
    for class_dir in class_dirs:
        files = sorted(
            file for file in class_dir.iterdir() if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        taken = 0
        for file in files:
            if not validate_image_file(file)[0]:
                continue
            samples.append(str(file.resolve()))
            taken += 1
            if len(samples) >= limit or taken >= max(1, limit // max(1, len(class_dirs))):
                break
        if len(samples) >= limit:
            return samples
    return samples
