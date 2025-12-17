# Image Workflow — Production Guidance

This document outlines steps and best practices for building a full image recognition pipeline using the toolkit.

Key recommendations:
- Use a larger dataset (thousands of images) and consistent image size (128–224)
- Use data augmentation (rotation, shift, zoom, brightness) during training
- Train on GPU (CUDA/cuDNN) for reasonable training time
- Save best checkpoints and evaluate on a held-out test set

Suggested training pipeline (high level):
1. Collect and organize images; create `labels.csv` with `filename,label,split`.
2. Create `tf.data` pipeline with augmentation and prefetching.
3. Use a deeper CNN (depth 4–6) with transfer learning (e.g., MobileNetV2) for better accuracy.
4. Train with callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.
5. Evaluate on a test split and calibrate thresholds if using anomaly-style detection.

Additional operational notes:
- When using the included Streamlit UI, the image tab supports a compact folder picker and a native OS folder dialog (when running locally) to choose the dataset folder.
- The UI defaults to saving models in the `.keras` format and writes a companion JSON metadata file containing `class_names` and configuration. Use the `Save model automatically` checkbox to persist the model on training completion, or use the `Save trained model` button to save the in-memory model afterwards.
Notes about Rust acceleration:
- Rust may help for heavy preprocessing loops (custom image transforms) via a Python extension, but not for TensorFlow training itself.
- Profile CPU-bound preprocessing before investing in Rust-based extensions.
