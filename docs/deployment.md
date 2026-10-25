# Deployment guide

## Local deployment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

- set the repository root as the app root
- define `streamlit run app.py` as the startup command
- keep the dataset out of the repo unless needed for a demo
- add a trusted, project-trained model artifact using an appropriate external storage option if the platform cannot host large files; do not load unverified model downloads
- document memory and CPU constraints clearly

## Model availability

The app reports model unavailability when no trained artifact is present and disables inference. It does not fabricate predictions or download unverified artifacts. When the dataset is available locally, users can select a genuine dataset image for inference.

## Hosting limitations

Community Cloud CPU, memory, and ephemeral disk constraints vary by plan. Training should be performed offline or in a suitable compute environment. SQLite prediction history is local to the instance and is not guaranteed to persist across redeployments.

## Offline field use and local contacts

The app does not download a model or call a hosted inference API. A trained Keras model can be exported with `python scripts/export_tflite.py`; the detection page uses `models/crop_model.tflite` if the Keras file is absent. Automated conversion and local inference tests use a synthetic model only. No trained CropShield artifact is currently available, so disease-level offline prediction and operation on a target field device have not been validated. Do not advertise or depend on field-validated offline inference until a trusted trained model and the complete workflow have been tested on the intended device. External source links require connectivity, and Streamlit must be served locally or over a reachable network.

Weather-based disease warnings remain unavailable unless a weather provider and a disease-specific verified rule with a citable source are both configured. The current deployment has neither. Severity scoring is omitted without manually annotated lesion-area validation. English browser read-aloud may be available; voice input and Kannada, Hindi, and Telugu translations are not enabled.

Administrators can add verified expert contacts to `data/local_expert_contacts.json`. The expected format is:

```json
{
  "schema_version": 1,
  "contacts": [
    {
      "name": "Verified contact name",
      "organization": "Verified agricultural extension organization",
      "url": "https://verified.example.org/contact",
      "phone": "Optional administrator-verified number"
    }
  ]
}
```

The values above are placeholders, not real contacts. Replace them only after independently confirming the organization, link, and phone number for the deployment region. The shipped file intentionally contains no contacts.

## Plant follow-up timeline privacy

The timeline stores plant names, observation notes, dates, and any selected prediction metadata in the local SQLite database. Photos are saved only after the user explicitly checks the retention-consent box and submits the observation; they are resized and stored separately under `data/plant_timeline_images/`. That directory is Git-ignored. Users can delete a single observation or a plant timeline to remove associated records and retained photos. Hosted deployments use local ephemeral storage unless separately configured, so records may not survive restarts or redeployments. Avoid storing personal or identifying details in plant names or notes.

Field coordinates are manually assigned grid positions and are not GPS. Repeated matching AI labels are described as “Possible spread risk,” never as confirmed spread. The expert-verification report is an export only; it is not submitted anywhere. A retained photo is added to the report only after a separate explicit selection.
