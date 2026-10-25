# Data directory

This folder is for local dataset storage and any project-specific sample data.

Do not commit large plant disease datasets or model weights to the repository unless they are intentionally tracked and below the size limits of your platform.

The project expects a PlantVillage-style directory structure with one class folder per disease group.

`disease_knowledge.json` is the structured store for India-specific, source-reviewed guidance. Classes without verified guidance use an explicit `unavailable` placeholder with no treatment claims or source URL. Do not add disease treatments until the source directly supports the claim and the guidance has been reviewed. Chemical product advice also requires a current local approval and label check.

`local_expert_contacts.json` is an administrator-managed list of verified local extension or university contacts. It is empty by design. Add only independently confirmed contact details and HTTPS URLs; do not infer or fabricate helplines.

`plant_timeline_images/` is created only when a user explicitly consents to retaining an observation photo. It is Git-ignored. Use the app's timeline deletion control to remove retained photos along with their observation records.

Field-grid coordinates are manually entered and are not GPS. Repeated AI labels are unconfirmed predictions, not confirmed disease spread. Action checklists must be derived only from `verified` knowledge records and preserve their source. The app currently has no configured weather provider or verified weather-disease rules; it must not generate weather-based risk advice without both. Expert reports are downloaded by the user and are never submitted automatically.

`detection_thresholds.json` contains editable display thresholds for the Disease Detection page's heuristic trust category. These values are not validated severity or diagnostic thresholds. OpenCV segmentation settings are implemented in `src/cropshield/image_quality.py`; the measured green-color coverage is only an approximation and should not be interpreted as verified leaf detection.

To prepare a downloaded ZIP archive, run:

```powershell
python scripts/prepare_dataset.py --dataset-archive "C:\path\to\PlantVillage.zip"
```

Only the archive's `color/` images are extracted to `data/PlantVillage-Dataset/color/`. The dataset remains local and is excluded from Git.
