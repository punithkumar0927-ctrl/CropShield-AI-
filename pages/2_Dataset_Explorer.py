from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cropshield.config import ARTIFACTS_DIR, MODELS_DIR
from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.dataset import (
    detect_dataset_root,
    inspect_dataset_integrity,
    iter_class_dirs,
    list_dataset_samples,
    summarize_dataset,
)

st.set_page_config(page_title="Dataset Explorer", page_icon="🌾", layout="wide")
st.title("Dataset Explorer")
st.caption("Explore the local PlantVillage dataset. Counts and charts are derived from files present on disk.")


@st.cache_data(ttl=300)
def get_dataset_root():
    return detect_dataset_root()


@st.cache_data(ttl=300)
def get_summary(dataset_path: str):
    return summarize_dataset(Path(dataset_path))


@st.cache_data(ttl=300)
def get_samples(dataset_path: str):
    return list_dataset_samples(Path(dataset_path), limit=12)


@st.cache_data(ttl=300)
def get_dataset_overview():
    """Read real train/validation/test split counts, model image size, and dataset source."""
    overview = {"split_counts": {}, "image_size": None, "source_url": None}
    for manifest_path in (ARTIFACTS_DIR / "dataset_split.csv", Path("artifacts/dataset_split.csv")):
        if manifest_path.is_file():
            try:
                manifest = pd.read_csv(manifest_path)
                if "split" in manifest.columns:
                    overview["split_counts"] = {
                        str(split): int(count) for split, count in manifest["split"].value_counts().items()
                    }
            except (OSError, ValueError):
                pass
            break
    config_path = MODELS_DIR / "model_config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                size = json.load(config_file).get("image_size")
            if isinstance(size, list) and len(size) == 2:
                overview["image_size"] = tuple(int(value) for value in size)
        except (OSError, ValueError, TypeError):
            pass
    readme_path = Path("data") / "README.md"
    if readme_path.is_file():
        text = readme_path.read_text(encoding="utf-8", errors="ignore")
        for token in ("https://github.com/spMohanty/PlantVillage-Dataset",):
            if token in text:
                overview["source_url"] = token
                break
    return overview


@st.cache_data(ttl=300)
def get_diverse_samples(dataset_path: str, per_class: int = 1, max_total: int = 10):
    """Pick one valid sample image per disease class so the gallery spans different conditions."""
    samples = []
    for class_dir in iter_class_dirs(Path(dataset_path)):
        files = sorted(
            file for file in class_dir.iterdir() if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for file in files[: per_class * 4]:
            try:
                from PIL import Image

                with Image.open(file) as image:
                    image.verify()
            except (OSError, ValueError):
                continue
            samples.append((str(file), class_dir.name))
            break
        if len(samples) >= max_total:
            break
    return samples


if st.sidebar.button("Refresh dataset"):
    get_dataset_root.clear()
    get_summary.clear()
    get_samples.clear()
    get_dataset_overview.clear()
    get_diverse_samples.clear()
    st.rerun()


dataset_root = get_dataset_root()
if dataset_root is None:
    st.warning(
        "PlantVillage dataset is not prepared locally. For a downloaded ZIP, run "
        "`python scripts/prepare_dataset.py --dataset-archive PATH`; for an extracted dataset, run "
        "`python scripts/prepare_dataset.py --dataset-root PATH`. The archive command extracts only the "
        "original color images."
    )
    stop_with_footer()

summary = get_summary(str(dataset_root))
counts = summary["class_counts"]
st.write(f"Dataset root: `{summary['dataset_root']}`")
st.metric("Class folders", summary["class_count"])
st.metric("Image files", sum(counts.values()))

# --- Dataset overview: source, model input size, and the real saved split counts. ---
st.subheader("Dataset overview")
overview = get_dataset_overview()
source_url = overview["source_url"] or "PlantVillage (spMohanty/PlantVillage-Dataset)"
if overview["source_url"]:
    st.markdown(f"**Dataset source:** [{source_url}]({overview['source_url']})")
else:
    st.markdown(f"**Dataset source:** {source_url}")
split_info = overview["split_counts"]
left_a, left_b, left_c, left_d, left_e = st.columns(5)
left_a.metric("Total images", sum(counts.values()))
left_b.metric("Number of classes", summary["class_count"])
left_c.metric("Model image size", f"{overview['image_size'][0]}×{overview['image_size'][1]}" if overview["image_size"] else "Not recorded")
if split_info:
    total_split = sum(split_info.values()) or 1
    left_d.metric("Train / Val / Test", " / ".join(str(split_info.get(name, 0)) for name in ("train", "validation", "test")))
    st.caption(
        "Deterministic stratified 70/15/15 split after exact-duplicate removal: "
        + ", ".join(f"{name} {split_info.get(name, 0) / total_split:.0%}" for name in ("train", "validation", "test"))
        + f" of {total_split} images."
    )
else:
    left_d.metric("Train / Val / Test", "Not split yet")
    st.caption("Run training to create the saved stratified split manifest.")
left_e.metric("Crops covered", len({name.split("___", maxsplit=1)[0] for name in counts}))
tomato_classes = sorted(
    class_name for class_name in counts if class_name.split("___", maxsplit=1)[0].casefold() == "tomato"
)
if tomato_classes:
    st.success(
        f"Focused training subset available: {len(tomato_classes)} Tomato classes. "
        "Run `python scripts/train_model.py --model-type mobilenet --crop Tomato` to train it."
    )

if counts:
    chart_frame = pd.DataFrame(
        [{"Class": class_name, "Images": count} for class_name, count in counts.items()]
    )
    chart = px.bar(
        chart_frame,
        x="Class",
        y="Images",
        title="Image count by class",
        color="Images",
        color_continuous_scale=["#d5e4cf", "#1d6849"],
    )
    chart.update_layout(xaxis_tickangle=-60)
    st.plotly_chart(chart, width="stretch")
    st.dataframe(chart_frame.sort_values("Images", ascending=False), width="stretch")
    imbalance_ratio = max(counts.values()) / min(counts.values()) if min(counts.values()) else None
    if imbalance_ratio is not None:
        st.write(f"Largest-to-smallest class count ratio: **{imbalance_ratio:.2f}**")
        if imbalance_ratio >= 2:
            st.warning(
                "The class distribution is imbalanced. Macro-averaged metrics and per-class recall help reveal "
                "minority-class performance that accuracy may obscure."
            )

sample_files = get_samples(str(dataset_root))
if sample_files:
    st.subheader("Real dataset samples")
    columns = st.columns(4)
    for index, sample in enumerate(sample_files):
        with columns[index % 4]:
            st.image(sample, caption=Path(sample).parent.name, width="stretch")

# --- Gallery across different disease classes: one sample per class, up to 10 classes. ---
st.subheader("Sample images from different disease classes")
diverse_samples = get_diverse_samples(str(dataset_root), per_class=1, max_total=10)
if diverse_samples:
    gallery_columns = st.columns(5)
    for index, (file_path, class_name) in enumerate(diverse_samples):
        with gallery_columns[index % 5]:
            st.image(file_path, caption=class_name, width="stretch")
    st.caption("First valid image per class folder, selected deterministically. Images remain on disk and are not copied.")

if st.button("Run image integrity audit"):
    with st.spinner("Checking image decodability, dimensions, and exact duplicate contents..."):
        audit = inspect_dataset_integrity(dataset_root)
    left, center, right = st.columns(3)
    left.metric("Valid images", audit["valid_images"])
    center.metric("Unreadable images", len(audit["invalid_images"]))
    right.metric("Exact duplicate files", len(audit["exact_duplicates"]))
    st.write("Image dimensions (width × height):")
    st.dataframe(
        pd.DataFrame(
            [{"Dimensions": dimension, "Images": count} for dimension, count in audit["dimensions"].items()]
        ),
        width="stretch",
    )
    if audit["invalid_images"]:
        st.warning(f"{len(audit['invalid_images'])} unreadable image file(s) were excluded from training.")
    if audit["exact_duplicates"]:
        st.info(
            "Exact duplicate files are excluded before the deterministic stratified train/validation/test split. "
            "Near-duplicates are not automatically detected."
        )

render_about_footer()
