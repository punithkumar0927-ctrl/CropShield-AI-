# CropShield AI Model Card

## Overview
CropShield AI is a crop disease classification project built around the PlantVillage dataset. The system is designed to classify leaf images into disease classes and communicate uncertainty rather than acting as a definitive field diagnosis tool.

## Intended use
Use the model for educational, research, and prototype demonstration tasks. It may support symptom triage in controlled image settings, but should not be used alone for treatment recommendations or real agronomic judgments without expert review.

## Dataset composition
The local PlantVillage color dataset contains 54,305 images across 38 class folders representing 14 crops. The image integrity audit found 54,305 valid images, zero unreadable images, and 21 exact duplicates. Exact duplicates are excluded before splitting. Exact labels and class counts are recorded in `data/class_labels.json` and `data/dataset_metadata.json`.

The deterministic seed-42 image-level split has 37,982 training, 8,126 validation, and 8,176 test images. The archive does not include the upstream leaf-group metadata, so related images may cross split boundaries. Results are not guaranteed to be leakage-free at the leaf level.

## Model architecture
The project includes a custom CNN baseline and a MobileNetV2 transfer-learning model. MobileNetV2 uses ImageNet pretrained weights, a frozen backbone, a 128-unit ReLU dense layer, dropout, and a 38-class softmax head. Input is resized to 224 × 224 RGB; MobileNetV2 preprocessing is embedded in the saved model graph. Training applies random horizontal flip, rotation, and zoom.

## Evaluation protocol
The workflow keeps a separate train/validation/test split, with validation used for checkpoint and confidence-threshold selection and the test set reserved for final reporting. Metrics include accuracy, macro and weighted precision/recall/F1, per-class results, confusion matrix, and CPU inference latency. Actual metrics are written to `reports/evaluation_metrics.json` only after evaluation runs. Model size and the evaluation device are included in the report.

## Results
No performance results should be claimed until `reports/evaluation_metrics.json` contains a completed held-out evaluation. High PlantVillage performance does not establish performance on field photos or unsupported diseases.

## Known limitations
- Public-dataset performance does not guarantee real-field performance.
- Different lighting, camera quality, background clutter, and leaf orientation can reduce accuracy.
- Imbalanced class distributions can lead to misleading accuracy values.
- Grad-CAM highlights regions that influence the prediction, but does not prove disease.
- The Smart Photo Coach uses basic statistical heuristics, not a verified leaf detector or disease-specific image-quality model.
- Entropy and prediction-margin alerts are caution signals, not a validated out-of-distribution detector.
- Disease treatment guidance remains unavailable for any class without a directly reviewed source. The current knowledge records explicitly mark all classes unavailable.

## Responsible use
Predictions should be treated as informational signals rather than definitive diagnosis. They should be validated with agronomic expertise and field inspection in real agricultural settings.

## Citation
Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science.
