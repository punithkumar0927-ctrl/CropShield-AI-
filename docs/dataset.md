# Dataset guide

## Download
Use the official PlantVillage dataset repository:

https://github.com/spMohanty/PlantVillage-Dataset

The upstream README currently describes 54,306 images across 14 crop species and 26 diseases, and links a Hugging Face dataset with predefined leaf-grouped train/test splits. Class-folder counts can differ by archive version and include healthy labels.

## License and split limitations

The upstream GitHub repository did not have a root `LICENSE` file when checked on 2026-10-24. The dataset is therefore not treated as having a confirmed redistribution license here; review the upstream dataset terms and research-paper attribution requirements before redistributing images or trained artifacts.

The supplied ZIP contains color images but no `leaf-map.json` or predefined split metadata. This project removes exact byte-for-byte duplicates and creates deterministic per-class splits, but it cannot guarantee that related images of the same leaf are kept together. For strict leaf-grouped evaluation, use the upstream Hugging Face split or provide the upstream `leaf-map.json` and add/validate a group-aware split before claiming leakage-free results. Current project metrics must be interpreted with this limitation.

## Prepare a ZIP archive

The project can selectively extract the original color images from the standard PlantVillage ZIP layout. It does not mix in grayscale or segmented images:

```powershell
python scripts/prepare_dataset.py --dataset-archive "C:\path\to\PlantVillage.zip"
```

This writes the folder-per-class dataset under `data/PlantVillage-Dataset/color/`. Local dataset files and ZIP archives are excluded from Git.
The supplied archive has 54,305 color images in 38 classes. The current duplicate scan identified 21 exact duplicates, which training excludes before splitting.

## Expected structure
The project expects a folder-per-class layout, for example:

```text
PlantVillage-Dataset/
  Apple___Apple_scab/
    0001.jpg
    0002.jpg
  Apple___Black_rot/
    ...
  ...
```

The dataset locator also recognizes common wrappers such as `raw/color`, `color`, `PlantVillage`, `PlantVillage-Dataset`, and `train`.

## Duplicate and quality checks
The dataset should be checked for:

- unreadable or corrupt images
- missing class folders
- exact duplicate contents (SHA-256)
- image dimensions above the configured safety limit

Use the Dataset Explorer's integrity audit for full image decoding and dimension checks. Training independently hashes valid image contents and excludes exact duplicates before splitting. Identical bytes assigned to different classes stop training as a labeling conflict. Near-duplicate images are not detected automatically and should be reviewed separately when building a production dataset.

Running `scripts/prepare_dataset.py` also writes `data/class_labels.json`, `data/dataset_metadata.json`, `data/image_quality_report.csv`, and a deterministic `data/split_manifest.csv`. It also adds explicit unavailable entries for classes that have no verified record in `data/disease_knowledge.json`; those entries contain no treatment advice. The quality report includes validity, dimensions, duplicate status, and simple brightness/focus/glare/color heuristics; these heuristics do not establish whether an image contains a plant or whether it is suitable for diagnosing a particular disease. Training writes its reproducible split manifest to both `data/split_manifest.csv` and `artifacts/dataset_split.csv`.

## Train/validation/test split
Training creates deterministic 70/15/15 image-level splits per class after exact duplicate removal. Each class needs at least three valid unique images. Validation data is used for checkpoint selection and confidence-threshold selection; test data is held out until the separate evaluation command. Test metrics are never used to select a model. Because the supplied ZIP lacks leaf-group metadata, related images may cross split boundaries; this split is not guaranteed to be leaf-group leakage-free.
