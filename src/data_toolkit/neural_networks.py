"""
Neural Networks Module for Data Analysis Toolkit

Provides deep learning models for:
1. MLP (Multi-Layer Perceptron) - Regression and Classification
2. LSTM - Time Series Forecasting
3. Autoencoder - Anomaly Detection

Uses TensorFlow/Keras backend with optional GPU acceleration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Dropout, BatchNormalization,
        Input, RepeatVector, TimeDistributed
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Neural network features disabled.")

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class NeuralNetworkModels:
    """
    Neural Network Models for Data Analysis
    
    Provides MLP, LSTM, and Autoencoder models with easy-to-use interface.
    
    Example:
        >>> nn = NeuralNetworkModels(df)
        >>> # MLP Regression
        >>> results = nn.mlp_regressor(features, target, epochs=100)
        >>> # LSTM Forecasting
        >>> forecast = nn.lstm_forecast('price', sequence_length=30, forecast_horizon=7)
        >>> # Autoencoder Anomaly Detection
        >>> anomalies = nn.autoencoder_anomaly_detection(features, contamination=0.05)
    """
    
    def __init__(self, df: pd.DataFrame = None):
        """Initialize with optional DataFrame."""
        self.df = df
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.label_encoders = {}
        
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for neural networks. "
                "Install with: pip install tensorflow"
            )
    
    def set_data(self, df: pd.DataFrame):
        """Set or update the DataFrame."""
        self.df = df
    
    # =========================================================================
    # MLP (Multi-Layer Perceptron)
    # =========================================================================
    
    def mlp_regressor(
        self,
        features: List[str],
        target: str,
        hidden_layers: List[int] = [64, 32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_size: float = 0.2,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Train MLP for regression tasks.
        
        Args:
            features: List of feature column names
            target: Target column name
            hidden_layers: List of neurons per hidden layer [64, 32, 16]
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout_rate: Dropout rate for regularization (0.0-0.5)
            learning_rate: Adam optimizer learning rate
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation during training
            test_size: Fraction for final test evaluation
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Training verbosity (0=silent, 1=progress, 2=detailed)
            
        Returns:
            Dictionary with model, metrics, predictions, and training history
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        X = self.df[features].values
        y = self.df[target].values
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        
        # Build model
        model = Sequential()
        model.add(Input(shape=(len(features),)))
        
        for i, units in enumerate(hidden_layers):
            model.add(Dense(units, activation=activation))
            model.add(BatchNormalization())
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))  # Output layer
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6
            ))
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate
        y_pred_scaled = model.predict(X_test, verbose=0).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        # Metrics
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # Store model
        self.models['mlp_regressor'] = model
        self.scalers['mlp_regressor'] = {'X': scaler_X, 'y': scaler_y}
        self.history['mlp_regressor'] = history.history
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'y_test': y_test_original,
            'predictions': y_pred,
            'training_history': history.history,
            'epochs_trained': len(history.history['loss']),
            'architecture': hidden_layers,
            'features': features,
            'target': target
        }
    
    def mlp_classifier(
        self,
        features: List[str],
        target: str,
        hidden_layers: List[int] = [64, 32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_size: float = 0.2,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Train MLP for classification tasks.
        
        Args:
            features: List of feature column names
            target: Target column name (categorical)
            hidden_layers: List of neurons per hidden layer
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            learning_rate: Adam optimizer learning rate
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation
            test_size: Fraction for test evaluation
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Training verbosity
            
        Returns:
            Dictionary with model, metrics, predictions, and confusion matrix
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        X = self.df[features].values
        y = self.df[target].values
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # One-hot encode for multi-class
        if n_classes > 2:
            y_onehot = to_categorical(y_encoded)
            loss = 'categorical_crossentropy'
            output_activation = 'softmax'
            output_units = n_classes
        else:
            y_onehot = y_encoded
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
            output_units = 1
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_onehot, test_size=test_size, random_state=42, stratify=y_encoded
        )
        y_test_labels = le.inverse_transform(
            np.argmax(y_test, axis=1) if n_classes > 2 else y_test.astype(int)
        )
        
        # Build model
        model = Sequential()
        model.add(Input(shape=(len(features),)))
        
        for units in hidden_layers:
            model.add(Dense(units, activation=activation))
            model.add(BatchNormalization())
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(output_units, activation=output_activation))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ))
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0)
        if n_classes > 2:
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred_encoded = (y_pred_proba.ravel() > 0.5).astype(int)
        y_pred_labels = le.inverse_transform(y_pred_encoded)
        
        # Metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        
        # Handle binary vs multi-class metrics
        avg = 'binary' if n_classes == 2 else 'weighted'
        precision = precision_score(y_test_labels, y_pred_labels, average=avg, zero_division=0)
        recall = recall_score(y_test_labels, y_pred_labels, average=avg, zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average=avg, zero_division=0)
        
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        
        # Store
        self.models['mlp_classifier'] = model
        self.scalers['mlp_classifier'] = {'X': scaler_X}
        self.label_encoders['mlp_classifier'] = le
        self.history['mlp_classifier'] = history.history
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classes': le.classes_.tolist(),
            'y_test': y_test_labels,
            'predictions': y_pred_labels,
            'prediction_probabilities': y_pred_proba,
            'training_history': history.history,
            'epochs_trained': len(history.history['loss']),
            'architecture': hidden_layers
        }
    
    def mlp_predict(
        self,
        new_data: pd.DataFrame,
        features: List[str],
        model_type: str = 'regressor'
    ) -> Dict[str, Any]:
        """
        Make predictions on new data using trained MLP model.
        
        Args:
            new_data: DataFrame with new samples to predict
            features: Feature column names (must match training features)
            model_type: 'regressor' or 'classifier'
            
        Returns:
            Dictionary with predictions and optional probabilities
        """
        model_key = f'mlp_{model_type}'
        
        if model_key not in self.models:
            return {'error': f'No trained {model_type} model found. Train a model first.'}
        
        model = self.models[model_key]
        scalers = self.scalers[model_key]
        
        # Check features exist
        missing_features = [f for f in features if f not in new_data.columns]
        if missing_features:
            return {'error': f'Missing features in new data: {missing_features}'}
        
        # Prepare data
        X_new = new_data[features].values
        X_scaled = scalers['X'].transform(X_new)
        
        # Predict
        predictions_raw = model.predict(X_scaled, verbose=0)
        
        if model_type == 'regressor':
            # Inverse transform for regression
            predictions = scalers['y'].inverse_transform(predictions_raw).ravel()
            return {
                'predictions': predictions,
                'n_samples': len(predictions),
                'model_type': 'regressor'
            }
        else:
            # Classification
            le = self.label_encoders[model_key]
            n_classes = len(le.classes_)
            
            if n_classes > 2:
                predicted_classes = np.argmax(predictions_raw, axis=1)
            else:
                predicted_classes = (predictions_raw.ravel() > 0.5).astype(int)
            
            predicted_labels = le.inverse_transform(predicted_classes)
            
            return {
                'predictions': predicted_labels,
                'probabilities': predictions_raw,
                'classes': le.classes_.tolist(),
                'n_samples': len(predicted_labels),
                'model_type': 'classifier'
            }
    
    # =========================================================================
    # LSTM (Long Short-Term Memory) for Time Series
    # =========================================================================
    
    def lstm_forecast(
        self,
        column: str,
        sequence_length: int = 30,
        forecast_horizon: int = 10,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        LSTM model for time series forecasting.
        
        Uses past 'sequence_length' values to predict next 'forecast_horizon' values.
        
        Args:
            column: Column name to forecast
            sequence_length: Number of past time steps to use as input
            forecast_horizon: Number of future steps to predict
            lstm_units: List of LSTM units per layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Adam optimizer learning rate
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Training verbosity
            
        Returns:
            Dictionary with model, metrics, forecasts, and history
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Get data
        data = self.df[column].dropna().values.reshape(-1, 1)
        
        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length - forecast_horizon + 1):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length:i + sequence_length + forecast_horizon].ravel())
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split (keep temporal order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential()
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences,
                              input_shape=(sequence_length, 1)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(forecast_horizon))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6
            ))
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_original = scaler.inverse_transform(y_test)
        y_pred_original = scaler.inverse_transform(y_pred_scaled)
        
        # Metrics (average over forecast horizon)
        mse = mean_squared_error(y_test_original.ravel(), y_pred_original.ravel())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original.ravel(), y_pred_original.ravel())
        
        # Generate future forecast
        last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        future_forecast_scaled = model.predict(last_sequence, verbose=0)
        future_forecast = scaler.inverse_transform(future_forecast_scaled).ravel()
        
        # Store model and parameters
        self.models['lstm_forecast'] = model
        self.scalers['lstm_forecast'] = scaler
        self.history['lstm_forecast'] = history.history
        self.history['lstm_sequence_length'] = sequence_length
        self.history['lstm_forecast_horizon'] = forecast_horizon
        self.history['lstm_column'] = column
        
        return {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'y_test': y_test_original,
            'predictions': y_pred_original,
            'future_forecast': future_forecast,
            'forecast_horizon': forecast_horizon,
            'sequence_length': sequence_length,
            'training_history': history.history,
            'epochs_trained': len(history.history['loss']),
            'column': column,
            'last_values': data[-sequence_length:].ravel()
        }
    
    def lstm_predict(
        self,
        new_data: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """
        Forecast using a trained LSTM model on new time series data.
        
        Args:
            new_data: DataFrame with the time series column
            column: Column name (must match training column)
            
        Returns:
            Dictionary with forecasts and metrics
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
            
        if 'lstm_forecast' not in self.models:
            return {'error': 'No trained LSTM model. Train first using lstm_forecast()'}
        
        model = self.models['lstm_forecast']
        scaler = self.scalers['lstm_forecast']
        
        # Get sequence_length and forecast_horizon from the stored history or defaults
        # These should match the training configuration
        sequence_length = self.history.get('lstm_sequence_length', 20)
        forecast_horizon = self.history.get('lstm_forecast_horizon', 10)
        
        if column not in new_data.columns:
            return {'error': f"Column '{column}' not found in new data"}
        
        # Get the time series data
        data = new_data[column].values.reshape(-1, 1)
        
        if len(data) < sequence_length:
            return {'error': f'Need at least {sequence_length} values (sequence_length), got {len(data)}'}
        
        # Scale the data using the training scaler
        data_scaled = scaler.transform(data)
        
        # Create sequences for prediction
        X = []
        for i in range(len(data_scaled) - sequence_length + 1):
            X.append(data_scaled[i:i + sequence_length])
        X = np.array(X)
        
        # Make predictions
        predictions_scaled = model.predict(X, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        
        # Generate future forecast from the last sequence
        last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        future_forecast_scaled = model.predict(last_sequence, verbose=0)
        future_forecast = scaler.inverse_transform(future_forecast_scaled).ravel()
        
        return {
            'predictions': predictions,
            'future_forecast': future_forecast,
            'n_samples': len(X),
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'column': column
        }
    
    def lstm_multivariate_forecast(
        self,
        features: List[str],
        target: str,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Multivariate LSTM: Use multiple features to forecast target.
        
        Args:
            features: List of feature columns (including target if desired)
            target: Target column to forecast
            sequence_length: Number of past time steps
            forecast_horizon: Steps ahead to predict
            lstm_units: LSTM units per layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation fraction
            early_stopping: Use early stopping
            patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Dictionary with model, metrics, and forecasts
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        all_cols = features if target in features else features + [target]
        data = self.df[all_cols].dropna().values
        target_idx = all_cols.index(target)
        
        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length - forecast_horizon + 1):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length:i + sequence_length + forecast_horizon, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential()
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences,
                              input_shape=(sequence_length, len(all_cols))))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(forecast_horizon))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform for target column
        # Create dummy array to inverse transform just the target
        dummy = np.zeros((len(y_test.ravel()), len(all_cols)))
        dummy[:, target_idx] = y_test.ravel()
        y_test_inv = scaler.inverse_transform(dummy)[:, target_idx]
        
        dummy[:, target_idx] = y_pred.ravel()
        y_pred_inv = scaler.inverse_transform(dummy)[:, target_idx]
        
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        
        self.models['lstm_multivariate'] = model
        self.scalers['lstm_multivariate'] = scaler
        
        return {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'y_test': y_test_inv,
            'predictions': y_pred_inv,
            'training_history': history.history,
            'epochs_trained': len(history.history['loss']),
            'features': all_cols,
            'target': target
        }
    
    # =========================================================================
    # Autoencoder for Anomaly Detection
    # =========================================================================
    
    def autoencoder_anomaly_detection(
        self,
        features: List[str],
        encoding_dim: int = None,
        hidden_layers: List[int] = None,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        contamination: float = 0.05,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Autoencoder-based anomaly detection.
        
        Trains autoencoder to reconstruct normal patterns. Points with high
        reconstruction error are flagged as anomalies.
        
        Args:
            features: List of feature columns
            encoding_dim: Dimension of the encoding layer (default: n_features // 2)
            hidden_layers: Custom encoder layers (default: auto-generated)
            activation: Activation function
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation fraction
            contamination: Expected fraction of anomalies (0.01-0.1)
            early_stopping: Use early stopping
            patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Dictionary with anomaly indices, scores, threshold, and model
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        X = self.df[features].dropna().values
        n_features = X.shape[1]
        
        # Default architecture
        if encoding_dim is None:
            encoding_dim = max(2, n_features // 3)
        
        if hidden_layers is None:
            # Auto-generate symmetric architecture
            hidden_layers = [n_features * 2, n_features, encoding_dim]
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Build autoencoder
        # Encoder
        input_layer = Input(shape=(n_features,))
        encoded = input_layer
        
        for units in hidden_layers:
            encoded = Dense(units, activation=activation)(encoded)
            encoded = BatchNormalization()(encoded)
            if dropout_rate > 0:
                encoded = Dropout(dropout_rate)(encoded)
        
        # Decoder (mirror architecture)
        decoded = encoded
        for units in reversed(hidden_layers[:-1]):
            decoded = Dense(units, activation=activation)(decoded)
            decoded = BatchNormalization()(decoded)
        
        # Output layer
        output_layer = Dense(n_features, activation='linear')(decoded)
        
        # Model
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Encoder model for embeddings
        encoder = Model(input_layer, encoded)
        
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ))
        
        # Train
        history = autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Compute reconstruction error
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Determine threshold based on contamination
        threshold = np.percentile(reconstruction_error, 100 * (1 - contamination))
        
        # Identify anomalies
        is_anomaly = reconstruction_error > threshold
        anomaly_indices = np.where(is_anomaly)[0].tolist()
        
        # Get encodings
        encodings = encoder.predict(X_scaled, verbose=0)
        
        # Store
        self.models['autoencoder'] = autoencoder
        self.models['encoder'] = encoder
        self.scalers['autoencoder'] = scaler
        self.history['autoencoder'] = history.history
        
        return {
            'autoencoder': autoencoder,
            'encoder': encoder,
            'anomaly_indices': anomaly_indices,
            'n_anomalies': len(anomaly_indices),
            'anomaly_percentage': len(anomaly_indices) / len(X) * 100,
            'reconstruction_errors': reconstruction_error,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'encodings': encodings,
            'encoding_dim': encoding_dim,
            'training_history': history.history,
            'epochs_trained': len(history.history['loss']),
            'features': features,
            'contamination': contamination
        }
    
    def predict_anomaly(self, new_data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Predict anomalies on new data using trained autoencoder.
        
        Args:
            new_data: New DataFrame to check for anomalies
            features: Feature columns (must match training features)
            
        Returns:
            Dictionary with anomaly predictions and scores
        """
        if 'autoencoder' not in self.models:
            return {'error': 'No autoencoder model trained. Run autoencoder_anomaly_detection first.'}
        
        autoencoder = self.models['autoencoder']
        scaler = self.scalers['autoencoder']
        
        X = new_data[features].values
        X_scaled = scaler.transform(X)
        
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Use stored threshold or compute new one
        threshold = getattr(self, '_anomaly_threshold', np.percentile(reconstruction_error, 95))
        is_anomaly = reconstruction_error > threshold
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_errors': reconstruction_error,
            'anomaly_indices': np.where(is_anomaly)[0].tolist(),
            'threshold': threshold
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_model_summary(self, model_name: str) -> str:
        """Get summary of a trained model."""
        if model_name not in self.models:
            return f"Model '{model_name}' not found."
        
        model = self.models[model_name]
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to file."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        self.models[model_name].save(filepath)
    
    def load_model(self, model_name: str, filepath: str):
        """Load a model from file."""
        self.models[model_name] = keras.models.load_model(filepath)
    
    def list_models(self) -> List[str]:
        """List all trained models."""
        return list(self.models.keys())


# Convenience function for quick use
def create_neural_network_analyzer(df: pd.DataFrame) -> NeuralNetworkModels:
    """Create a NeuralNetworkModels instance with data."""
    return NeuralNetworkModels(df)
