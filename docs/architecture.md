# CropShield AI architecture

## High-level structure

The project is organized into:

- app and Streamlit pages for user interactions
- data utilities for dataset detection and image validation
- model training code for baseline and transfer-learning networks
- evaluation utilities for classification metrics and confusion matrices
- explainability utilities for Grad-CAM
- local SQLite history for minimal prediction tracking

## Processing flow

1. User uploads an image.
2. Input validation screens for type, size, and readability.
3. The image is normalized to the expected training size.
4. The selected TensorFlow model predicts class probabilities.
5. The app surfaces the top prediction and confidence. It also stores minimal metadata in SQLite.
6. Grad-CAM can explain the model's focus area when a valid convolutional model is available.

## Capture-quality and field-risk checks

Before inference, the Smart Photo Coach reports image dimensions, brightness, a Laplacian-variance focus heuristic, bright-glare fraction, aspect ratio, and the fraction of green-colored pixels. Very small, extremely dark/bright, or extremely low-detail images are blocked for retake; other findings produce actionable warnings. The green-pixel statistic is not a plant detector, and these thresholds have not been validated as a disease-quality classifier.

The optional field-risk checklist scores only user-reported affected-leaf proportion, nearby symptomatic plants, prolonged leaf wetness, and overhead irrigation. It displays each contributing factor and labels the result as preliminary. It is not a disease diagnosis, outbreak forecast, or treatment recommendation.

Prediction ambiguity is additionally flagged using normalized prediction entropy and the margin between the two highest class probabilities. These are caution signals, not a validated out-of-distribution detector; the app warns users that unsupported conditions may not be covered by the model.

## Data responsibilities

- `dataset.py`: locate and inspect dataset structure
- `preprocessing.py`: enforce validation and prepare arrays for inference
- `train.py`: train/fine-tune models
- `evaluate.py`: report metrics and model comparison
- `predict.py`: perform inference for single images
- `explainability.py`: produce heatmaps for model interpretation
- `history.py`: maintain a compact prediction log
- `history.py`: maintain opt-in plant observations and explicitly retained local follow-up photos
- field grid, provenance labels, local verified-action checklists, and non-diagnostic repeated-prediction summaries
- `expert_request.py`: build an HTML report for the user to review and export; no submission endpoint is configured
- `export_tflite.py` and `TFLiteModel`: convert a trained Keras artifact and run local CPU inference from the resulting one-input/one-output TFLite model
- `image_quality.py`: capture statistics, OpenCV HSV/contour leaf-like coverage estimate, and optional largest-region crop
- `trust.py`: configurable heuristic trust category using image quality, confidence, top-two margin, validation calibration, and optional Grad-CAM overlap

## Innovation feature limits

The field grid uses user-entered row and column coordinates. A repeated same-class label is only a pattern in AI predictions and is not confirmed disease spread. Checklists are created only from records marked verified and retain their cited source. Expert exports explicitly state that no heatmap is available and include a retained photo only with a separate user selection. Weather risk is withheld unless weather data and a matching verified rule are both available; neither is currently configured. TensorFlow Lite conversion and CPU inference are implemented and tested with a synthetic model, but no trained CropShield model is available, so field use remains unvalidated. No lesion-area severity score, voice input, or Kannada/Hindi/Telugu translation is available. English read-aloud uses the browser speech API when supported.

Detection accepts JPG/JPEG/PNG/WEBP images with a 10 MB size ceiling and a 224 x 224 minimum before inference. OpenCV segmentation is a green-color/contour estimate only. It can miss yellow/brown leaves and is not a semantic leaf detector. High trust is not issued unless a validation-derived confidence threshold exists; all trust categories remain heuristic. The opt-in prediction history stores class/probability metadata, file name, timing, quality measurements, and user note but no image bytes. Grad-CAM is available only for compatible Keras models; the TFLite inference path reports it unavailable.
