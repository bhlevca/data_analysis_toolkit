"""
Biomass Segmentation Module
Contains U-Net based semantic segmentation for ecological biomass analysis:
- Dreissena mussels (zebra/quagga mussels)
- Cladophora algae
- General aquatic vegetation

Uses TensorFlow/Keras for U-Net architecture with transfer learning support.

Version: 1.0
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import VGG16, ResNet50
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import image processing libraries
try:
    from PIL import Image
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class BiomassSegmentation:
    """
    U-Net based semantic segmentation for aquatic biomass.
    
    Supports multi-class segmentation for:
    - Class 0: Background (water, substrate)
    - Class 1: Dreissena mussels (zebra/quagga)
    - Class 2: Cladophora algae
    - Class 3: Other vegetation (optional)
    
    Features:
    - U-Net with encoder-decoder architecture
    - Optional VGG16/ResNet encoder for transfer learning
    - Data augmentation for limited datasets
    - Dice loss for imbalanced classes
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3),
                 n_classes: int = 3, encoder: str = 'standard'):
        """
        Initialize BiomassSegmentation model.
        
        Args:
            input_shape: (height, width, channels) for input images
            n_classes: Number of segmentation classes
            encoder: 'standard', 'vgg16', or 'resnet50' for encoder backbone
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder = encoder
        self.model = None
        self.history = None
        
        # Class labels
        self.class_labels = {
            0: 'Background',
            1: 'Dreissena (Mussels)',
            2: 'Cladophora (Algae)',
            3: 'Other Vegetation'
        }
        
        # Colors for visualization (RGBA)
        self.class_colors = [
            (0, 0, 0, 0),       # Background - transparent
            (255, 0, 0, 180),   # Dreissena - red
            (0, 255, 0, 180),   # Cladophora - green
            (0, 0, 255, 180)    # Other - blue
        ]
    
    def build_unet(self) -> Model:
        """
        Build standard U-Net architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (contracting path)
        # Block 1
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        p1 = layers.Dropout(0.1)(p1)
        
        # Block 2
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        p2 = layers.Dropout(0.1)(p2)
        
        # Block 3
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D(2)(c3)
        p3 = layers.Dropout(0.2)(p3)
        
        # Block 4
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
        c4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D(2)(c4)
        p4 = layers.Dropout(0.2)(p4)
        
        # Bridge (bottleneck)
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Dropout(0.3)(c5)
        
        # Decoder (expanding path)
        # Block 6
        u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Dropout(0.2)(c6)
        
        # Block 7
        u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Dropout(0.2)(c7)
        
        # Block 8
        u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Dropout(0.1)(c8)
        
        # Block 9
        u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
        c9 = layers.BatchNormalization()(c9)
        
        # Output layer
        outputs = layers.Conv2D(self.n_classes, 1, activation='softmax')(c9)
        
        model = Model(inputs, outputs)
        return model
    
    def build_vgg_unet(self) -> Model:
        """
        Build U-Net with VGG16 encoder (transfer learning).
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained VGG16
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze encoder layers initially
        for layer in vgg.layers:
            layer.trainable = False
        
        # Get encoder layers
        inputs = vgg.input
        c1 = vgg.get_layer('block1_conv2').output  # 64 filters
        c2 = vgg.get_layer('block2_conv2').output  # 128 filters
        c3 = vgg.get_layer('block3_conv3').output  # 256 filters
        c4 = vgg.get_layer('block4_conv3').output  # 512 filters
        c5 = vgg.get_layer('block5_conv3').output  # 512 filters
        
        # Decoder
        u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
        
        u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
        
        u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
        
        outputs = layers.Conv2D(self.n_classes, 1, activation='softmax')(c9)
        
        model = Model(inputs, outputs)
        return model
    
    def dice_loss(self, y_true, y_pred, smooth: float = 1e-6):
        """
        Dice loss for segmentation (handles class imbalance better than cross-entropy).
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice loss value
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def combined_loss(self, y_true, y_pred):
        """
        Combined Dice + Cross-Entropy loss.
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(y_true, y_pred)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return dice + tf.reduce_mean(ce)
    
    def build_model(self, learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Build and compile the segmentation model.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with model info
        """
        try:
            if self.encoder == 'vgg16':
                self.model = self.build_vgg_unet()
            else:
                self.model = self.build_unet()
            
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=self.combined_loss,
                metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.n_classes)]
            )
            
            return {
                'status': 'success',
                'encoder': self.encoder,
                'input_shape': self.input_shape,
                'n_classes': self.n_classes,
                'n_parameters': self.model.count_params(),
                'summary': self._get_model_summary()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not built"
        
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 8,
              augment: bool = True) -> Dict[str, Any]:
        """
        Train the segmentation model.
        
        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training masks (N, H, W, n_classes) one-hot encoded
            X_val: Validation images
            y_val: Validation masks
            epochs: Number of training epochs
            batch_size: Batch size
            augment: Whether to use data augmentation
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            return {'error': 'Model not built. Call build_model() first.'}
        
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Data augmentation
            if augment:
                data_gen = self._create_augmentation()
                train_gen = data_gen.flow(X_train, y_train, batch_size=batch_size)
                steps_per_epoch = len(X_train) // batch_size
            else:
                train_gen = None
                steps_per_epoch = None
            
            # Train
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            else:
                validation_data = None
            
            if train_gen:
                self.history = self.model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                self.history = self.model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=1
                )
            
            return {
                'status': 'success',
                'epochs_trained': len(self.history.history['loss']),
                'final_loss': float(self.history.history['loss'][-1]),
                'final_accuracy': float(self.history.history['accuracy'][-1]),
                'history': {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_augmentation(self):
        """Create data augmentation generator."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            fill_mode='reflect'
        )
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Input image (H, W, C) or (N, H, W, C)
            
        Returns:
            Dictionary with predicted mask and class probabilities
        """
        if self.model is None:
            return {'error': 'Model not built or trained'}
        
        try:
            # Ensure batch dimension
            if image.ndim == 3:
                image = np.expand_dims(image, 0)
            
            # Normalize if needed
            if image.max() > 1:
                image = image / 255.0
            
            # Resize if needed
            if image.shape[1:3] != self.input_shape[:2]:
                resized = []
                for img in image:
                    if CV2_AVAILABLE:
                        resized.append(cv2.resize(img, self.input_shape[:2]))
                    else:
                        resized.append(np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(self.input_shape[:2])) / 255.0)
                image = np.array(resized)
            
            # Predict
            predictions = self.model.predict(image, verbose=0)
            
            # Get class masks
            class_masks = np.argmax(predictions, axis=-1)
            
            # Calculate class areas
            class_areas = {}
            for i in range(self.n_classes):
                count = np.sum(class_masks == i)
                total = class_masks.size
                class_areas[self.class_labels.get(i, f'Class {i}')] = {
                    'pixels': int(count),
                    'percentage': float(count / total * 100)
                }
            
            return {
                'mask': class_masks,
                'probabilities': predictions,
                'class_areas': class_areas
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_biomass(self, image: np.ndarray, pixel_size_mm: float = 1.0) -> Dict[str, Any]:
        """
        Analyze biomass coverage in an image.
        
        Args:
            image: Input image
            pixel_size_mm: Size of each pixel in millimeters (for area calculation)
            
        Returns:
            Dictionary with biomass analysis results
        """
        prediction = self.predict(image)
        if 'error' in prediction:
            return prediction
        
        mask = prediction['mask']
        if mask.ndim == 3:
            mask = mask[0]
        
        total_pixels = mask.size
        pixel_area_mm2 = pixel_size_mm ** 2
        total_area_mm2 = total_pixels * pixel_area_mm2
        
        results = {
            'total_area_mm2': float(total_area_mm2),
            'total_area_cm2': float(total_area_mm2 / 100),
            'image_size': mask.shape,
            'classes': {}
        }
        
        for i in range(self.n_classes):
            class_name = self.class_labels.get(i, f'Class {i}')
            class_pixels = np.sum(mask == i)
            class_area_mm2 = class_pixels * pixel_area_mm2
            
            results['classes'][class_name] = {
                'pixels': int(class_pixels),
                'coverage_percent': float(class_pixels / total_pixels * 100),
                'area_mm2': float(class_area_mm2),
                'area_cm2': float(class_area_mm2 / 100)
            }
        
        # Biomass-specific metrics
        dreissena_coverage = results['classes'].get('Dreissena (Mussels)', {}).get('coverage_percent', 0)
        cladophora_coverage = results['classes'].get('Cladophora (Algae)', {}).get('coverage_percent', 0)
        
        results['summary'] = {
            'total_biomass_coverage': float(dreissena_coverage + cladophora_coverage),
            'dreissena_coverage': float(dreissena_coverage),
            'cladophora_coverage': float(cladophora_coverage),
            'dreissena_to_cladophora_ratio': float(dreissena_coverage / cladophora_coverage) if cladophora_coverage > 0 else float('inf')
        }
        
        return results
    
    def visualize_prediction(self, image: np.ndarray, 
                            show_overlay: bool = True,
                            alpha: float = 0.5) -> 'plt.Figure':
        """
        Visualize segmentation prediction.
        
        Args:
            image: Input image
            show_overlay: Whether to overlay prediction on original image
            alpha: Transparency for overlay
            
        Returns:
            Matplotlib figure
        """
        if not MPL_AVAILABLE:
            return None
        
        prediction = self.predict(image)
        if 'error' in prediction:
            return None
        
        mask = prediction['mask']
        if mask.ndim == 3:
            mask = mask[0]
        if image.ndim == 4:
            image = image[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if image.max() > 1:
            image = image / 255.0
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        cmap = ListedColormap(['black', 'red', 'green', 'blue'][:self.n_classes])
        axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=self.n_classes - 1)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        if show_overlay:
            # Create colored overlay
            overlay = np.zeros((*mask.shape, 4))
            for i in range(self.n_classes):
                color = np.array(self.class_colors[i]) / 255.0
                overlay[mask == i] = color
            
            axes[2].imshow(image)
            axes[2].imshow(overlay, alpha=alpha)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        else:
            # Show class probabilities
            probs = prediction['probabilities'][0] if prediction['probabilities'].ndim == 4 else prediction['probabilities']
            axes[2].imshow(np.max(probs, axis=-1), cmap='viridis')
            axes[2].set_title('Prediction Confidence')
            axes[2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Dreissena'),
            Patch(facecolor='green', label='Cladophora'),
        ]
        if self.n_classes > 3:
            legend_elements.append(Patch(facecolor='blue', label='Other'))
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str) -> Dict[str, Any]:
        """Save the model to disk."""
        if self.model is None:
            return {'error': 'No model to save'}
        
        try:
            self.model.save(filepath)
            return {'status': 'success', 'path': filepath}
        except Exception as e:
            return {'error': str(e)}
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load a model from disk."""
        try:
            self.model = tf.keras.models.load_model(
                filepath, 
                custom_objects={
                    'dice_loss': self.dice_loss,
                    'combined_loss': self.combined_loss
                }
            )
            return {'status': 'success', 'path': filepath}
        except Exception as e:
            return {'error': str(e)}


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess an image for segmentation.
    
    Args:
        image_path: Path to image file
        target_size: (height, width) to resize to
        
    Returns:
        Preprocessed image array
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV not available. Install with: pip install opencv-python")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image.astype(np.float32)


def create_sample_dataset(n_samples: int = 10, 
                         image_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic sample dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        image_size: (height, width) of images
        
    Returns:
        Tuple of (images, masks)
    """
    h, w = image_size
    images = []
    masks = []
    
    for _ in range(n_samples):
        # Create random image
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Create mask with random blobs
        mask = np.zeros((h, w), dtype=np.int32)
        
        # Add random "mussel" blobs (class 1)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(20, w-20), np.random.randint(20, h-20)
            r = np.random.randint(10, 30)
            y, x = np.ogrid[:h, :w]
            blob = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
            mask[blob] = 1
        
        # Add random "algae" blobs (class 2)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(20, w-20), np.random.randint(20, h-20)
            r = np.random.randint(15, 40)
            y, x = np.ogrid[:h, :w]
            blob = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
            mask[blob] = 2
        
        images.append(img / 255.0)
        
        # One-hot encode mask
        mask_onehot = np.zeros((h, w, 3))
        for c in range(3):
            mask_onehot[:, :, c] = (mask == c).astype(np.float32)
        masks.append(mask_onehot)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)
