from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropshield.dataset import detect_dataset_root, iter_class_dirs  # noqa: E402
from src.cropshield.train import train_model  # noqa: E402


def select_training_classes(
    dataset_root: Path,
    crop: str,
    requested_classes: list[str] | None = None,
) -> list[str] | None:
    if requested_classes:
        return requested_classes
    if crop.casefold() == "all":
        return None
    selected = [
        directory.name
        for directory in iter_class_dirs(dataset_root)
        if directory.name.split("___", maxsplit=1)[0].casefold() == crop.casefold()
    ]
    if len(selected) < 2:
        raise ValueError(
            f"Crop '{crop}' has fewer than two class folders in {dataset_root}. "
            "Use --classes with exact folder names or --crop all."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CropShield AI models.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to the dataset root.")
    parser.add_argument("--model-type", type=str, default="mobilenet", choices=["baseline", "mobilenet"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--crop",
        default="Tomato",
        help="Train classes belonging to this crop (default: Tomato); use 'all' for every class.",
    )
    selection.add_argument(
        "--classes",
        nargs="+",
        help="Train an explicit list of exact class-folder names.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else detect_dataset_root()
    if dataset_root is None:
        raise FileNotFoundError("Dataset not found. Download the PlantVillage dataset and use --dataset-root.")
    selected_classes = select_training_classes(dataset_root, args.crop, args.classes)
    result = train_model(
        dataset_root=str(dataset_root),
        model_type=args.model_type,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_names=selected_classes,
    )
    print(f"Model type: {result['model_type']}")
    print(f"Classes trained: {len(result['labels'])}")
    print(f"Labels: {result['labels'][:10]}...")


if __name__ == "__main__":
    main()
