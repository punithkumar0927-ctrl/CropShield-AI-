# CropShield AI 2.0

CropShield AI 2.0 is an intelligent crop disease detection platform built for portfolio and learning use. It demonstrates a complete ML workflow: dataset preparation, model training, evaluation, explainability, and a Streamlit dashboard for real image inference.

## Project overview

This project is designed for job applications and portfolio interviews in machine learning, data science, and computer vision roles. The goal is to classify crop leaf diseases from images using a deep learning model, provide transparent evaluation metrics, and present the workflow in an understandable interactive web app.

## Problem statement

Early detection of plant disease can lower crop losses and improve agricultural decision-making. However, many crop diseases look visually similar and are difficult to distinguish without specialist review. This project builds a reproducible image-classification workflow to support crop disease detection from leaf photographs.

## Key features

- Streamlit dashboard with multiple pages
- Real image upload and prediction workflow
- PlantVillage dataset integration instructions
- Baseline CNN and MobileNetV2 transfer learning models
- Classification evaluation with confusion matrices and per-class metrics
- Explainable AI using Grad-CAM
- Prediction history stored locally in SQLite
- Per-plant follow-up timeline with dated notes and optional, explicitly consented local photo retention
- Named plots with a simple user-entered field grid, distinct user-observation/AI labels, and non-diagnostic repeated-prediction warnings
- Verified-source recovery checklists stored locally and an export-only expert verification report (nothing is submitted automatically)
- Weather-risk messaging that withholds assessment unless both real weather data and a matching verified disease rule are available
- Browser text-to-speech for verified English guidance where supported
- Validation-derived uncertainty threshold; predictions remain preliminary and can be flagged as uncertain
- Smart Photo Coach for resolution, brightness, focus, glare, and framing feedback
- OpenCV color/contour-based leaf-like coverage estimates, multiple-region feedback, and optional experimental crop comparison
- Validation-aware High/Medium/Low/Cannot Assess trust labels; high trust is withheld without a validation-derived model threshold
- Grad-CAM overlay and a leaf-like-region focus warning for compatible Keras CNNs
- Optional transparent field-risk checklist based only on user-reported conditions
- Prioritized action plan with safety precautions and expert escalation
- Printable HTML prediction report
- Disease guidance is displayed only when its India-specific record has been verified against an authoritative source
- Editable local expert-contact configuration with no fabricated contact details
- Tests using pytest
- CI workflow for GitHub Actions

## Architecture overview

```text
User upload -> Streamlit app -> preprocessing -> trained TensorFlow model -> prediction -> metrics + explainability
                                    |
                                    +-> dataset utilities + evaluation scripts + history storage
```

## Screenshots

Add screenshots after running the app locally and renaming files to `docs/screenshots/...`.

## Dataset source and license

This project is designed for the PlantVillage dataset:

- Source: https://github.com/spMohanty/PlantVillage-Dataset
- License: Please review the upstream dataset license and attribution requirements before redistributing data or trained artifacts.
- Use only relevant folders and respect the dataset copyright notices.

## Installation

Use Python 3.13 for the pinned TensorFlow runtime.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset setup

The included PlantVillage ZIP archive can be prepared directly. The script extracts only the original color images into `data/PlantVillage-Dataset/color/` and leaves grayscale/segmented variants untouched:

```powershell
python scripts/prepare_dataset.py --dataset-archive "archive (10).zip"
```

The included archive contains 54,305 color images across 38 classes. Exact duplicate checks found 21 duplicate images; training excludes these from its split.

For an already extracted dataset, validate its root with:

```powershell
python scripts/prepare_dataset.py --dataset-root "C:\path\to\PlantVillage-Dataset"
```

The prepared local data is excluded from Git. If the dataset is not available, the app reports that status rather than generating predictions or metrics.
Preparation also writes local class labels, dataset metadata, an image-by-image integrity/quality report, and the reproducible split manifest under `data/`. Training writes its split manifest to both `data/split_manifest.csv` and `artifacts/dataset_split.csv`.

## Training

The split is generated reproducibly by class after exact duplicate images are removed. The supplied ZIP has no leaf-group metadata, so related leaves may cross split boundaries; see [docs/dataset.md](docs/dataset.md) for this evaluation limitation and the upstream grouped-split alternative. Training selects checkpoints using validation loss; the held-out test split is not accessed for checkpoint selection. By default, training starts with the 10 Tomato classes rather than all 38 classes. Use `--crop all` to opt into every class, or `--classes` to provide exact class-folder names.

```powershell
python scripts/train_model.py --dataset-root "data\PlantVillage-Dataset\color" --model-type mobilenet --crop Tomato --batch-size 16 --image-size 224 --epochs 8
```

## Evaluation

After training, evaluate the saved held-out test split and start the app with:

```powershell
python scripts/train_model.py
python scripts/evaluate_model.py
streamlit run app.py
```

For a specific local dataset or architecture, pass `--dataset-root "data\PlantVillage-Dataset\color"` and, when needed, `--model-type baseline` or `--model-type mobilenet`. Evaluation reads `artifacts/dataset_split.csv` (falling back to `data/split_manifest.csv`) and does not create a new test split.

```powershell
python scripts/evaluate_model.py --dataset-root "data\PlantVillage-Dataset\color" --model-type mobilenet
```

Each successful evaluation writes actual held-out results to `reports/evaluation_metrics.json`, the compatibility copy to `artifacts/evaluation_metrics.json`, per-class metrics to `reports/classification_report.json`, and the confusion matrix to `reports/confusion_matrix.json` and `reports/confusion_matrix.png`. The side-by-side report is saved to `artifacts/model_comparison.json`. Evaluation reports compare models but do not select one based on test results. Training histories and the split manifest are also written under `artifacts/` and `reports/`.

## Run the Streamlit app

```powershell
streamlit run app.py
```

## Testing

Install the development tools once:

```powershell
pip install -r requirements-dev.txt
```

Then run the focused lint checks and test suite:

```powershell
ruff check app.py src pages scripts tests --select E4,E7,E9,F
pytest -q
```
## Run Locally

After starting the Streamlit app, open:

**http://localhost:8501/**

> This link works only while the app is running on your own computer.  
> It is not a public hosted URL.

## Model evaluation table

This table is populated only after running real evaluation. Do not add hardcoded metrics.

| Model | Accuracy | Macro F1 | Weighted F1 | Status |
|---|---:|---:|---:|---|
| Baseline CNN | Not evaluated yet | Not evaluated yet | Not evaluated yet | Pending |
| MobileNetV2 | Not evaluated yet | Not evaluated yet | Not evaluated yet | Pending |

## Deployment

The project is prepared for Streamlit Community Cloud. Use the following checklist:

- ensure `requirements.txt` is present
- set the app entry point to `app.py`
- keep the model artifact outside the repo unless explicitly managed as a downloadable asset
- document storage and CPU limits
- the app will report that predictions are unavailable if no trained model is present; it does not fabricate a demo model or output
- use the dataset-sample selector only after adding the dataset locally

## Responsible use

Predictions are informational only and are not definitive agricultural diagnoses. Model outputs should be reviewed by trained growers, agronomists, or plant pathologists before making crop treatment decisions.

## Disease guidance and reports

The detection page can download a printable HTML report with the prediction, confidence, top model outputs, and any verified guidance available for that class. The **Treatment & Prevention Guide** page also lists supported classes. Reports and pages state clearly when no reviewed guidance is available; model labels are never converted into treatment instructions automatically.

Curated records belong in `data/disease_knowledge.json`. Each record must be keyed by the exact model class label and include symptoms, look-alikes, management, prevention, safety precautions, a credible HTTPS source, and a review date. The current knowledge file contains no verified disease records, so treatment advice is intentionally unavailable until India-specific sources are reviewed. Chemical products and application details must not be added without confirmation against current Indian approval and the product label.

Dataset preparation now creates an unavailable, no-advice placeholder for any supported class lacking a knowledge record. These placeholders are explicitly unverified, contain no treatment claims or sources, and exist only to make missing coverage visible.

New training runs derive an uncertainty threshold from validation predictions, selecting the largest validation coverage that reaches at least 90% accuracy on accepted validation examples. The threshold and calibration results are saved in the model configuration. This is not an out-of-distribution detector: unfamiliar plants or field images can still produce misleading scores.

The photo coach uses image statistics and OpenCV color segmentation. Leaf coverage, contour counts, cropped-image comparison, and the Grad-CAM leaf-focus check are experimental, color-based indicators; they do not identify a leaf or diagnose image suitability. A categorical trust heuristic combines these signals with model confidence, top-two margin, and any validation-derived confidence threshold. These signals are not a validated out-of-distribution detector, and High trust is withheld when model validation calibration is unavailable. The optional field-risk checklist is a transparent screening aid and not a disease forecast.

Detection accepts JPG, JPEG, PNG, and WEBP up to 10 MB and requires at least 224 x 224 pixels before inference. Prediction history is opt-in and stores the image filename, top predictions, trust category, timing, quality measurements, and optional user notes locally; image bytes are not stored in history. Timeline photo storage remains a separate, explicit opt-in.

`data/detection_thresholds.json` holds the editable heuristic cutoffs for the displayed trust category. They are not a clinically or agronomically validated score. Grad-CAM is supported only for compatible Keras CNNs; TensorFlow Lite models do not currently expose the attention heatmap. The optional leaf-like crop comparison is experimental and never replaces the original-image prediction.

The application uses local model files and local guidance when available. A trained `.keras` model can be exported to TensorFlow Lite, and the detection page will use `models/crop_model.tflite` when the Keras model is absent:

```powershell
python scripts/export_tflite.py
```

The conversion and local prediction path are covered by an automated synthetic-model test, but this repository has no trained CropShield model artifact, so offline field accuracy and device operation have not been validated. Do not rely on this build as a field-validated offline tool. Linked source pages require internet. Weather-based risk assessment, Kannada/Hindi/Telugu treatment translations, voice input, and severity scoring are not enabled; no severity score is shown without annotated lesion-area validation. English read-aloud uses the browser's speech engine where supported and does not translate the guidance. Configure verified local experts in `data/local_expert_contacts.json`; the shipped file is intentionally empty.

## Plant follow-up timeline

Open **Plant Follow-up Timeline** to create a locally tracked plant or plot and add dated notes, optional model labels, and optional confidence values. The Disease Detection page can also add a successful scan to a selected timeline after explicit opt-in. These repeated observations help organize field visits; score changes do not prove that disease progressed or treatment worked. Timeline metadata is stored in the local SQLite history database. Follow-up photos are not saved unless the photo-retention checkbox is explicitly selected and the observation is submitted; retained copies are resized JPEGs stored under the ignored `data/plant_timeline_images/` directory. Delete a single observation or its entire timeline to remove associated records and retained photos. Do not store unnecessary personal information. Cloud-hosted storage is not guaranteed to persist between redeployments.

Field-map coordinates are manually entered grid positions, not GPS. “Possible spread risk” means the same non-healthy model label appeared in multiple AI scans in the named plot; it is not a confirmed spread event. User observations, user-reported expert confirmations, and AI predictions are labeled separately; expert-confirmed entries are not independently authenticated by CropShield. The treatment guide only saves recovery steps from verified knowledge-base records and records each checklist's source. Expert reports are generated for download in the browser and are never transmitted by the app; location and retained-photo inclusion are optional. No weather provider or verified weather-disease rule is currently configured, so the app explicitly withholds weather risk assessments.

## GitHub repository

- Repository URL: Add your GitHub repository URL here after publishing the project.

## Model card

See [reports/model_card.md](reports/model_card.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
