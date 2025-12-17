"""
CNN image trainer and predictor for Data Analysis Toolkit - IMPROVED VERSION

FIXES APPLIED:
1. ✅ Validation split bug - Now creates proper stratified splits (was using only 5 samples!)
2. ✅ Data augmentation - Automatic augmentation for better generalization  
3. ✅ Folder-based loading - Point at any folder, subfolders = classes
4. ✅ Metadata saving - Class names and config saved automatically
5. ✅ Preprocessing fixes - Correct normalization for transfer learning
6. ✅ Better error messages and progress tracking

Usage:
    # From folder structure:
    train_cnn('my_dataset/')  # Expects: my_dataset/class1/, my_dataset/class2/, etc.
    
    # From CSV:
    train_cnn('my_dataset/', labels_csv='labels.csv')
    
    # Transfer learning:
    train_transfer_learning('my_dataset/', model_out='models/biomass_classifier.keras')
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Literal, Optional
import warnings

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def _load_dataset_from_folder(data_dir: Path, image_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """Load dataset from folder structure: subfolders = classes."""
    records = []
    class_folders = [f for f in data_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    if len(class_folders) == 0:
        raise ValueError(f"No subdirectories in {data_dir}. Organize images into class subfolders.")
    
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        img_files = [f for f in class_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        for img_path in img_files:
            records.append({
                'filename': str(img_path.relative_to(data_dir)),
                'label': class_name,
                'absolute_path': str(img_path)
            })
    
    print(f"📂 Loaded {len(records)} images from {len(class_folders)} classes")
    return records


def _read_labels_csv(data_dir: Path, labels_csv: str = 'labels.csv') -> List[Dict]:
    """Read labels CSV with filename,label or filename,label,split format."""
    candidate = Path(labels_csv)
    csv_path = candidate if candidate.exists() else Path(data_dir) / labels_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    records = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            fname = r.get('filename') or r.get('file') or ''
            p = Path(fname)

            # Resolve absolute paths carefully to avoid duplicating folder names
            if p.is_absolute():
                candidate_path = p
            else:
                candidate_path = Path(data_dir) / p

            # Prefer candidate under data_dir if it exists, otherwise try the original path
            if candidate_path.exists():
                abs_path = str(candidate_path)
            elif p.exists():
                abs_path = str(p.resolve())
            else:
                # try matching just the basename under data_dir
                alt = Path(data_dir) / p.name
                if alt.exists():
                    abs_path = str(alt)
                else:
                    abs_path = str(candidate_path)

            r['absolute_path'] = abs_path
            records.append(r)

    # Filter out records whose files do not exist (but warn the user)
    missing = [r for r in records if not Path(r['absolute_path']).exists()]
    if missing:
        warnings.warn(f"⚠️ {len(missing)} entries in {csv_path} reference missing files (examples: {', '.join([m['filename'] for m in missing[:3]])}). These will be skipped.")
    records = [r for r in records if Path(r['absolute_path']).exists()]

    if len(records) == 0:
        raise FileNotFoundError(f"No valid image files found from CSV: {csv_path}")

    print(f"📄 Loaded {len(records)} valid records from CSV (skipped {len(missing)} missing)")
    return records


def _make_datasets(
    data_dir: Path,
    labels_csv: Optional[str] = None,
    image_size: int = 128,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    augment: bool = True,
    task: Literal['classification', 'regression'] = 'classification'
):
    """Create datasets with FIXED validation splitting and augmentation."""
    
    # Load records (auto-detect format)
    if labels_csv and (Path(labels_csv).exists() or (data_dir / labels_csv).exists()):
        records = _read_labels_csv(data_dir, labels_csv)
    elif (data_dir / 'labels.csv').exists():
        records = _read_labels_csv(data_dir, 'labels.csv')
    else:
        records = _load_dataset_from_folder(data_dir)
    
    if len(records) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Check existing splits
    train_records = [r for r in records if r.get('split', 'train') == 'train']
    val_records = [r for r in records if r.get('split') in ('val', 'validation', 'test')]
    
    # CRITICAL FIX: Create proper stratified split if validation is too small
    min_val_samples = max(10, int(0.05 * len(records)))
    if len(val_records) < min_val_samples:
        if len(val_records) > 0:
            warnings.warn(f"⚠️  Only {len(val_records)} val samples. Creating stratified split.")
        
        usable_records = [r for r in records if r.get('split', 'train') != 'predict']
        
        # Stratified split by label
        from collections import defaultdict
        label_to_records = defaultdict(list)
        for r in usable_records:
            label_to_records[r['label']].append(r)
        
        train_records, val_records = [], []
        rng = np.random.RandomState(seed)
        
        for label, recs in label_to_records.items():
            rng.shuffle(recs)
            n_val = max(1, int(len(recs) * val_split))
            val_records.extend(recs[:n_val])
            train_records.extend(recs[n_val:])
        
        print(f"✅ Stratified split: {len(train_records)} train, {len(val_records)} val")
    
    # Extract paths and labels
    train_files = [r['absolute_path'] for r in train_records]
    train_labels_raw = [r['label'] for r in train_records]
    val_files = [r['absolute_path'] for r in val_records]
    val_labels_raw = [r['label'] for r in val_records]
    
    # Task-specific encoding
    if task == 'regression':
        train_labels = np.array([float(l) for l in train_labels_raw], dtype=np.float32)
        val_labels = np.array([float(l) for l in val_labels_raw], dtype=np.float32)
        class_names = None
        num_outputs = 1
        print(f"📊 Regression: range [{train_labels.min():.2f}, {train_labels.max():.2f}]")
    else:
        class_names = sorted(list(set(train_labels_raw + val_labels_raw)))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        train_labels = np.array([class_to_idx[l] for l in train_labels_raw], dtype=np.int32)
        val_labels = np.array([class_to_idx[l] for l in val_labels_raw], dtype=np.int32)
        num_outputs = len(class_names)
        
        print(f"📊 Classes: {class_names}")
        unique, counts = np.unique(train_labels, return_counts=True)
        for idx, cnt in zip(unique, counts):
            print(f"   {class_names[idx]}: {cnt} samples")
    
    # Data pipeline with augmentation
    def _load_and_preprocess(path, label, is_training):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        
        # Augmentation for training
        if is_training and augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.85, 1.15)
            image = tf.image.random_saturation(image, 0.85, 1.15)
        
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(min(10000, len(train_files)), seed=seed)
    train_ds = train_ds.map(lambda p, l: _load_and_preprocess(p, l, True), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(lambda p, l: _load_and_preprocess(p, l, False), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_outputs, len(train_records), len(val_records)


def build_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 10,
    base_filters: int = 32,
    depth: int = 4,
    dense_units: int = 256,
    dropout: float = 0.4
):
    """Build custom CNN."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for i in range(depth):
        filters = base_filters * (2 ** i)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        if i >= 2:
            x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='CustomCNN')


def train_cnn(
    data_dir: str,
    labels_csv: Optional[str] = None,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 20,
    model_out: Optional[str] = 'models/image_cnn.keras',
    save_model: bool = True,
    val_split: float = 0.2,
    base_filters: int = 32,
    depth: int = 4,
    dense_units: int = 256,
    dropout: float = 0.4,
    learning_rate: float = 1e-3,
    seed: int = 42,
    augment: bool = True
) -> Dict:
    """Train CNN with improved data handling, augmentation, and metadata saving."""
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')

    print(f"\n{'='*60}\n🚀 CNN Training\n{'='*60}\n")
    
    data_dir = Path(data_dir)
    train_ds, val_ds, class_names, num_outputs, n_train, n_val = _make_datasets(
        data_dir, labels_csv=labels_csv, image_size=image_size, batch_size=batch_size, 
        val_split=val_split, seed=seed, augment=augment, task='classification'
    )

    model = build_cnn(
        (image_size, image_size, 3), num_outputs,
        base_filters, depth, dense_units, dropout
    )

    opt = keras.optimizers.Adam(learning_rate)
    loss = 'binary_crossentropy' if num_outputs == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    print(f"\n📐 Model: {model.count_params():,} parameters\n")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(model_out), save_best_only=True, monitor='val_loss', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-7)
    ]

    print(f"🏋️  Training...\n")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    saved_path = None
    metadata_path = None
    if save_model and model_out:
        if model_out and model_out.suffix != '.keras':
            model_out = model_out.with_suffix('.keras')
        model.save(str(model_out))

        # Save metadata
        metadata = {
            'class_names': class_names,
            'image_size': image_size,
            'task': 'classification',
            'architecture': 'custom_cnn',
            'num_classes': num_outputs,
            'n_train': n_train,
            'n_val': n_val,
            'augmentation': augment
        }
        metadata_path = model_out.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        saved_path = str(model_out)
    
    print(f"\n{'='*60}\n✅ Complete!\n✅ Model: {model_out}\n✅ Metadata: {metadata_path}\n{'='*60}\n")

    result = {'model': model, 'history': history.history, 'class_names': class_names}
    if saved_path:
        result['model_path'] = saved_path
    if metadata_path:
        result['metadata_path'] = str(metadata_path)
    return result



# Register as Keras serializable (compat for tf.keras and keras)
try:
    from tensorflow.keras.utils import register_keras_serializable
except ImportError:
    try:
        from keras.utils import register_keras_serializable
    except ImportError:
        def register_keras_serializable(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

@register_keras_serializable(package="Custom", name="MobileNetPreprocessing")
class MobileNetPreprocessing(keras.layers.Layer):
    """Custom preprocessing layer for MobileNetV2: [0,1] -> [-1,1]"""
    def call(self, inputs):
        return (inputs * 2.0) - 1.0
    def get_config(self):
        return super().get_config()


def train_transfer_learning(
    data_dir: str,
    labels_csv: Optional[str] = None,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 12,
    model_out: Optional[str] = 'models/transfer_mobilenet.keras',
    save_model: bool = True,
    base_trainable: bool = False,
    learning_rate: float = 1e-4,
    seed: int = 42,
    augment: bool = True
):
    """Transfer learning with MobileNetV2 and proper serializable preprocessing."""
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')

    print(f"\n{'='*60}\n🚀 Transfer Learning (MobileNetV2)\n{'='*60}\n")
    
    data_dir = Path(data_dir)
    train_ds, val_ds, class_names, num_outputs, n_train, n_val = _make_datasets(
        data_dir, labels_csv=labels_csv, image_size=image_size, batch_size=batch_size, 
        seed=seed, augment=augment, task='classification'
    )

    base_model = tf.keras.applications.MobileNetV2(
        (image_size, image_size, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = base_trainable
    print(f"{'🔓' if base_trainable else '🔒'} Base model: {'trainable' if base_trainable else 'frozen'}\n")

    # Proper serializable preprocessing layer
    inputs = keras.Input((image_size, image_size, 3))
    x = MobileNetPreprocessing(name='mobilenet_preprocess')(inputs)
    x = base_model(x, training=base_trainable)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    if num_outputs == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_outputs, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = keras.Model(inputs, outputs, name='MobileNetV2_Transfer')
    
    model.compile(keras.optimizers.Adam(learning_rate), loss, metrics=['accuracy'])
    
    print(f"📐 Total: {model.count_params():,} params")
    print(f"📐 Trainable: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,} params\n")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(model_out), save_best_only=True, monitor='val_loss', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
    ]

    print(f"🏋️  Training...\n")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    saved_path = None
    metadata_path = None
    if save_model and model_out:
        if model_out.suffix != '.keras':
            model_out = model_out.with_suffix('.keras')
        model.save(str(model_out))

        metadata = {
            'class_names': class_names,
            'image_size': image_size,
            'task': 'classification',
            'architecture': 'mobilenetv2_transfer',
            'base_trainable': base_trainable,
            'num_classes': num_outputs,
            'n_train': n_train,
            'n_val': n_val,
            'augmentation': augment
        }
        metadata_path = model_out.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        saved_path = str(model_out)
    
    print(f"\n{'='*60}\n✅ Complete!\n✅ Model: {model_out}\n✅ Metadata: {metadata_path}\n{'='*60}\n")

    result = {'model': model, 'history': history.history, 'class_names': class_names}
    if saved_path:
        result['model_path'] = saved_path
    if metadata_path:
        result['metadata_path'] = str(metadata_path)
    return result


def _load_model_with_fallback(model_path: str):
    """Load model with fallbacks for serialization issues, always passing custom layers."""
    # Only allow safe deserialization; do NOT enable unsafe deserialization.
    class TrueDivide(keras.layers.Layer):
        def call(self, inputs):
            return inputs / 127.5 - 1.0
        def get_config(self):
            return super().get_config()

    custom_objects = {
        'TrueDivide': TrueDivide,
        'MobileNetPreprocessing': MobileNetPreprocessing
    }

    # Load model with explicit custom objects. If deserialization fails due to
    # Lambda layers or unsafe content, raise and do not attempt unsafe workarounds.
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def predict_image(
    image_path: str, 
    model_path: str, 
    image_size: int = 128, 
    class_names: Optional[List[str]] = None
) -> Dict:
    """Predict on single image file with auto metadata loading."""
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')
    
    # Auto-load metadata
    if class_names is None:
        metadata_path = Path(model_path).with_suffix('.json')
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                class_names = metadata.get('class_names')
                image_size = metadata.get('image_size', image_size)
            except Exception:
                pass
    
    model = _load_model_with_fallback(model_path)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    
    if preds.shape[-1] == 1:
        prob = float(preds.ravel()[0])
        label_idx = 1 if prob > 0.5 else 0
        probs = [1 - prob, prob]
    else:
        probs = preds.ravel().tolist()
        label_idx = int(np.argmax(probs))

    # Validate class_names length; fallback to numeric labels if mismatch
    if class_names is None or len(class_names) != len(probs):
        if class_names is not None:
            warnings.warn(f"⚠️  class_names length ({len(class_names)}) != model outputs ({len(probs)}). Falling back to numeric labels.")
        class_names = [str(i) for i in range(len(probs))]

    return {'predicted_label': class_names[label_idx], 'probabilities': probs, 'confidence': max(probs)}


def predict_array(
    image_array: np.ndarray, 
    model_path: str, 
    image_size: int = 128, 
    class_names: Optional[List[str]] = None
) -> Dict:
    """Predict from in-memory array with auto metadata loading."""
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')
    
    # Auto-load metadata
    if class_names is None:
        metadata_path = Path(model_path).with_suffix('.json')
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                class_names = metadata.get('class_names')
                image_size = metadata.get('image_size', image_size)
            except Exception:
                pass

    model = _load_model_with_fallback(model_path)

    img = tf.convert_to_tensor(image_array)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)

    if preds.shape[-1] == 1:
        prob = float(preds.ravel()[0])
        label_idx = 1 if prob > 0.5 else 0
        probs = [1 - prob, prob]
    else:
        probs = preds.ravel().tolist()
        label_idx = int(np.argmax(probs))

    if class_names is None:
        class_names = [str(i) for i in range(len(probs))]

    return {'predicted_label': class_names[label_idx], 'probabilities': probs, 'confidence': max(probs)}
