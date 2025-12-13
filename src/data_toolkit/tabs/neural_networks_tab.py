"""
Tab module for the Data Analysis Toolkit
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from neural_networks import NeuralNetworkModels, TF_AVAILABLE

PLOTLY_TEMPLATE = "plotly_white"

def render_neural_networks_tab():
    """Render Neural Networks tab for deep learning models"""
    st.header("🧠 Neural Networks")
    
    # Check TensorFlow availability
    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow is not installed. Neural Networks features are unavailable.")
        st.info("Install TensorFlow with: `pip install tensorflow`")
        return
    
    # Educational introduction
    with st.expander("ℹ️ About Neural Networks - Deep Learning for Data Analysis", expanded=False):
        st.markdown("""
        ### 🧠 Neural Network Models
        
        This module provides deep learning models for:
        
        **1. Multi-Layer Perceptron (MLP)**
        - Flexible feedforward networks for regression and classification
        - Configurable hidden layers and activation functions
        - Dropout regularization to prevent overfitting
        
        **2. LSTM (Long Short-Term Memory)**
        - Specialized for time series forecasting
        - Captures long-range temporal dependencies
        - Generates future predictions beyond your data
        
        **3. Autoencoder**
        - Unsupervised anomaly detection
        - Learns compressed data representations
        - Detects anomalies via reconstruction error
        
        **When to use Neural Networks:**
        - Complex non-linear relationships
        - Large datasets (1000+ samples recommended)
        - Time series with temporal patterns
        - When traditional ML models underperform
        """)
    
    if st.session_state.df is None:
        st.warning("📁 Please load data first using the Data & Preview tab")
        return
    
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for neural network analysis")
        return
    
    # Model selection
    nn_model_type = st.selectbox(
        "Select Neural Network Model",
        ["MLP (Multi-Layer Perceptron)", "LSTM (Time Series Forecasting)", "Autoencoder (Anomaly Detection)"]
    )
    
    # Initialize NeuralNetworkModels
    nn = NeuralNetworkModels(df)
    
    # =========================================================================
    # MLP
    # =========================================================================
    if nn_model_type == "MLP (Multi-Layer Perceptron)":
        st.subheader("🔮 Multi-Layer Perceptron")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target Variable", numeric_cols, key="mlp_target")
            task_type = st.radio("Task Type", ["regression", "classification"], key="mlp_task")
        
        with col2:
            feature_cols = st.multiselect(
                "Feature Variables",
                [c for c in numeric_cols if c != target_col],
                default=[c for c in numeric_cols if c != target_col][:5],
                key="mlp_features"
            )
        
        # Architecture settings
        with st.expander("🔧 Network Architecture", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                n_layers = st.slider("Hidden Layers", 1, 5, 2, key="mlp_layers")
                neurons = st.slider("Neurons per Layer", 8, 256, 64, key="mlp_neurons")
            with col2:
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, key="mlp_dropout")
                activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="mlp_activation")
            with col3:
                epochs = st.slider("Epochs", 10, 200, 50, key="mlp_epochs")
                batch_size = st.slider("Batch Size", 8, 128, 32, key="mlp_batch")
        
        hidden_layers = [neurons] * n_layers
        
        # Filter out target from features if accidentally selected (can happen with cached widget state)
        feature_cols_clean = [f for f in feature_cols if f != target_col]
        
        # Show warning if target was in features
        if len(feature_cols_clean) < len(feature_cols):
            st.warning(f"⚠️ Removed '{target_col}' from features (target cannot be a feature)")
        
        if st.button("🚀 Train MLP", type="primary", key="train_mlp"):
            if len(feature_cols_clean) < 1:
                st.error("Select at least 1 feature (excluding target)")
                return
            
            with st.spinner("Training MLP..."):
                try:
                    if task_type == "regression":
                        results = nn.mlp_regressor(
                            features=feature_cols_clean,
                            target=target_col,
                            hidden_layers=hidden_layers,
                            activation=activation,
                            dropout_rate=dropout,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                    else:
                        results = nn.mlp_classifier(
                            features=feature_cols_clean,
                            target=target_col,
                            hidden_layers=hidden_layers,
                            activation=activation,
                            dropout_rate=dropout,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                    
                    # Store trained model info in session state for prediction
                    st.session_state['nn_model'] = nn
                    st.session_state['nn_features'] = feature_cols_clean
                    st.session_state['nn_target'] = target_col
                    st.session_state['nn_task'] = task_type
                    st.session_state['nn_training_results'] = results
                    st.session_state['nn_trained'] = True
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ MLP Training failed: {str(e)}")
        
        # Display training results if model was trained (persists after file upload)
        if st.session_state.get('nn_trained', False) and 'nn_training_results' in st.session_state:
            results = st.session_state['nn_training_results']
            stored_task = st.session_state.get('nn_task', 'regression')
            
            st.success("✅ MLP Training Complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            if stored_task == "regression":
                col1.metric("Test RMSE", f"{results['rmse']:.4f}")
                col2.metric("Test MAE", f"{results['mae']:.4f}")
                col3.metric("Test R²", f"{results['r2']:.4f}")
            else:
                col1.metric("Test Accuracy", f"{results['accuracy']:.2%}")
                col2.metric("Test Precision", f"{results['precision']:.2%}")
                col3.metric("Test F1 Score", f"{results['f1_score']:.2%}")
            
            # Training history
            if 'training_history' in results:
                st.subheader("📈 Training History")
                history = results['training_history']
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Metric"])
                
                # Loss
                fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='red')), row=1, col=1)
                
                # Metric
                metric_key = 'mae' if stored_task == 'regression' else 'accuracy'
                if metric_key in history:
                    fig.add_trace(go.Scatter(y=history[metric_key], name=f'Train {metric_key}', line=dict(color='green')), row=1, col=2)
                    if f'val_{metric_key}' in history:
                        fig.add_trace(go.Scatter(y=history[f'val_{metric_key}'], name=f'Val {metric_key}', line=dict(color='orange')), row=1, col=2)
                
                fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("✅ Model trained! Now upload a prediction file below to test on new data.")
        
        # Prediction on new data
        st.markdown("---")
        st.subheader("🔮 Predict & Compare on New Data")
        
        # Check if we have a properly trained model with stored features
        has_trained_model = 'nn_model' in st.session_state and 'nn_features' in st.session_state
        
        if has_trained_model:
            stored_features = st.session_state['nn_features']
            stored_target = st.session_state['nn_target']
            stored_task = st.session_state['nn_task']
            
            st.caption(f"Upload a CSV with features ({', '.join(stored_features[:3])}{'...' if len(stored_features) > 3 else ''}) AND the target column '{stored_target}' to compare predictions vs actual values")
            st.info(f"**Trained model expects:** Features: {stored_features}, Target: '{stored_target}', Task: {stored_task}")
        else:
            st.info("💡 Train a model above first, then upload a prediction file here.")
            stored_features = feature_cols  # Fallback for display only
            stored_target = target_col
            stored_task = task_type
        
        predict_file = st.file_uploader("Upload Prediction/Test File (CSV)", type=['csv'], key="mlp_predict_file")
        
        if predict_file is not None:
            try:
                predict_df = pd.read_csv(predict_file)
                st.write(f"📊 Loaded {len(predict_df)} samples with {len(predict_df.columns)} columns")
                st.dataframe(predict_df.head(), use_container_width=True)
                
                # Check if we have a trained model
                if not has_trained_model:
                    st.warning("⚠️ Train a model first before making predictions")
                else:
                    # Check features exist in prediction file (use stored features from training)
                    missing_features = [f for f in stored_features if f not in predict_df.columns]
                    has_target = stored_target in predict_df.columns
                    
                    if missing_features:
                        st.error(f"❌ Missing features in prediction file: {missing_features}")
                        st.info(f"Expected features (from training): {stored_features}")
                    elif not has_target:
                        st.warning(f"⚠️ Target column '{stored_target}' not found in prediction file. Will show predictions only (no comparison).")
                    
                    if st.button("🎯 Make Predictions", type="primary", key="mlp_predict_btn"):
                        if missing_features:
                            st.error("Cannot predict - missing features")
                        else:
                            with st.spinner("Making predictions..."):
                                try:
                                    nn_model = st.session_state['nn_model']
                                    pred_results = nn_model.mlp_predict(
                                        new_data=predict_df,
                                        features=stored_features,
                                        model_type='regressor' if stored_task == 'regression' else 'classifier'
                                    )
                                    
                                    if 'error' in pred_results:
                                        st.error(f"❌ {pred_results['error']}")
                                    else:
                                        st.success(f"✅ Predictions complete for {pred_results['n_samples']} samples!")
                                        
                                        # Add predictions to dataframe with clear column name
                                        result_df = predict_df.copy()
                                        result_df[f'Predicted_{stored_target}'] = pred_results['predictions']
                                        
                                        # If we have actual values, compare
                                        if has_target:
                                            actual = predict_df[stored_target].values
                                            predicted = pred_results['predictions']
                                            
                                            if stored_task == 'regression':
                                                # Calculate metrics
                                                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                                mse = mean_squared_error(actual, predicted)
                                                mae = mean_absolute_error(actual, predicted)
                                                r2 = r2_score(actual, predicted)
                                                
                                                st.subheader("📊 Prediction Performance")
                                                col1, col2, col3 = st.columns(3)
                                                col1.metric("RMSE", f"{np.sqrt(mse):.4f}")
                                                col2.metric("MAE", f"{mae:.4f}")
                                                col3.metric("R²", f"{r2:.4f}")
                                                
                                                # Actual vs Predicted plot (by sample index)
                                                st.subheader(f"🎯 Actual vs Predicted Values for '{stored_target}'")
                                                fig = go.Figure()
                                                
                                                sample_indices = list(range(len(actual)))
                                                
                                                # Actual values
                                                fig.add_trace(go.Scatter(
                                                    x=sample_indices,
                                                    y=actual,
                                                    mode='markers+lines',
                                                    marker=dict(color='blue', size=8),
                                                    line=dict(color='blue', width=1),
                                                    name=f'Actual {stored_target}',
                                                    hovertemplate='Sample: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                                                ))
                                                
                                                # Predicted values
                                                fig.add_trace(go.Scatter(
                                                    x=sample_indices,
                                                    y=predicted,
                                                    mode='markers+lines',
                                                    marker=dict(color='red', size=8, symbol='diamond'),
                                                    line=dict(color='red', width=1, dash='dot'),
                                                    name=f'Predicted {stored_target}',
                                                    hovertemplate='Sample: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
                                                ))
                                                
                                                fig.update_layout(
                                                    title=f'Actual vs Predicted: {stored_target}',
                                                    xaxis_title='Sample Index',
                                                    yaxis_title=stored_target,
                                                    template=PLOTLY_TEMPLATE,
                                                    height=500,
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Residuals plot
                                                residuals = predicted - actual
                                                result_df['Residual'] = residuals
                                                
                                                fig2 = go.Figure()
                                                fig2.add_trace(go.Scatter(
                                                    x=sample_indices,
                                                    y=residuals,
                                                    mode='markers',
                                                    marker=dict(color='coral', size=8, opacity=0.7),
                                                    name='Residuals'
                                                ))
                                                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                                                fig2.update_layout(
                                                    title='Residuals Plot (Predicted - Actual)',
                                                    xaxis_title='Sample Index',
                                                    yaxis_title='Residual',
                                                    template=PLOTLY_TEMPLATE,
                                                    height=400
                                                )
                                                st.plotly_chart(fig2, use_container_width=True)
                                                
                                            else:  # Classification
                                                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                                                accuracy = accuracy_score(actual, predicted)
                                                st.subheader("📊 Classification Performance")
                                                st.metric("Accuracy", f"{accuracy:.2%}")
                                                # Classification report
                                                report = classification_report(actual, predicted, output_dict=True)
                                                st.dataframe(pd.DataFrame(report).T, use_container_width=True)

                                                # Confusion matrix plot
                                                st.subheader("🔲 Confusion Matrix")
                                                cm = confusion_matrix(actual, predicted)
                                                classes = np.unique(np.concatenate([actual, predicted]))
                                                fig_cm = go.Figure(data=go.Heatmap(
                                                    z=cm,
                                                    x=[f'Pred: {c}' for c in classes],
                                                    y=[f'Actual: {c}' for c in classes],
                                                    colorscale='Blues',
                                                    showscale=True,
                                                    text=cm,
                                                    texttemplate='%{text}',
                                                    textfont={"size": 14}
                                                ))
                                                fig_cm.update_layout(
                                                    title='Confusion Matrix',
                                                    xaxis_title='Predicted',
                                                    yaxis_title='Actual',
                                                    template=PLOTLY_TEMPLATE,
                                                    height=400
                                                )
                                                st.plotly_chart(fig_cm, use_container_width=True)

                                                # Bar plot of class distributions
                                                st.subheader("📊 Class Distribution: Actual vs Predicted")
                                                actual_counts = pd.Series(actual).value_counts().sort_index()
                                                pred_counts = pd.Series(predicted).value_counts().sort_index()
                                                all_classes = sorted(set(actual_counts.index) | set(pred_counts.index))
                                                fig_bar = go.Figure()
                                                fig_bar.add_trace(go.Bar(
                                                    name='Actual',
                                                    x=[str(c) for c in all_classes],
                                                    y=[actual_counts.get(c, 0) for c in all_classes],
                                                    marker_color='steelblue'
                                                ))
                                                fig_bar.add_trace(go.Bar(
                                                    name='Predicted',
                                                    x=[str(c) for c in all_classes],
                                                    y=[pred_counts.get(c, 0) for c in all_classes],
                                                    marker_color='crimson'
                                                ))
                                                fig_bar.update_layout(
                                                    title='Class Distribution: Actual vs Predicted',
                                                    xaxis_title='Class',
                                                    yaxis_title='Count',
                                                    barmode='group',
                                                    template=PLOTLY_TEMPLATE,
                                                    height=400
                                                )
                                                st.plotly_chart(fig_bar, use_container_width=True)
                                        
                                        # Show results table
                                        st.subheader("📋 Results Table")
                                        st.dataframe(result_df, use_container_width=True)
                                        
                                        # Download button
                                        csv = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Predictions",
                                            data=csv,
                                            file_name="nn_predictions.csv",
                                            mime="text/csv"
                                        )
                                except Exception as e:
                                    st.error(f"❌ Prediction failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to load prediction file: {str(e)}")
    
    # =========================================================================
    # LSTM
    # =========================================================================
    elif nn_model_type == "LSTM (Time Series Forecasting)":
        st.subheader("📈 LSTM Time Series Forecasting")
        
        col1, col2 = st.columns(2)
        with col1:
            ts_column = st.selectbox("Time Series Column", numeric_cols, key="lstm_column")
            sequence_length = st.slider("Sequence Length (lookback)", 5, 100, 20, key="lstm_seq")
        
        with col2:
            forecast_horizon = st.slider("Forecast Horizon", 1, 50, 10, key="lstm_horizon")
            epochs = st.slider("Epochs", 10, 200, 50, key="lstm_epochs")
        
        with st.expander("🔧 LSTM Architecture", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                lstm_units = st.slider("LSTM Units", 16, 256, 64, key="lstm_units")
                n_lstm_layers = st.slider("LSTM Layers", 1, 4, 2, key="lstm_n_layers")
            with col2:
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, key="lstm_dropout")
                batch_size = st.slider("Batch Size", 8, 64, 16, key="lstm_batch")
        
        if st.button("🚀 Train LSTM", type="primary", key="train_lstm"):
            with st.spinner("Training LSTM model..."):
                try:
                    results = nn.lstm_forecast(
                        column=ts_column,
                        sequence_length=sequence_length,
                        forecast_horizon=forecast_horizon,
                        lstm_units=[lstm_units] * n_lstm_layers,
                        dropout_rate=dropout,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    # Store model info for prediction
                    st.session_state['lstm_model'] = nn
                    st.session_state['trained_lstm_column'] = ts_column
                    st.session_state['trained_lstm_seq_length'] = sequence_length
                    st.session_state['trained_lstm_forecast_horizon'] = forecast_horizon
                    st.session_state['lstm_training_results'] = results
                    st.session_state['lstm_trained'] = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ LSTM Training failed: {str(e)}")
        
        # Display training results if model was trained (persists after file upload)
        if st.session_state.get('lstm_trained', False) and 'lstm_training_results' in st.session_state:
            results = st.session_state['lstm_training_results']
            stored_column = st.session_state.get('trained_lstm_column', ts_column)
            
            st.success("✅ LSTM Training Complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Test RMSE", f"{results['rmse']:.4f}")
            col2.metric("Test MAE", f"{results['mae']:.4f}")
            col3.metric("Epochs Trained", results['epochs_trained'])
            
            # Forecast plot
            st.subheader("📊 Forecast Results")
            
            fig = go.Figure()
            
            # Actual values
            actual = results['y_test'].ravel()
            fig.add_trace(go.Scatter(
                y=actual,
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Predicted values
            predicted = results['predictions'].ravel()
            fig.add_trace(go.Scatter(
                y=predicted,
                name='Predicted',
                line=dict(color='green')
            ))
            
            # Future forecast
            forecast = results['future_forecast']
            forecast_x = list(range(len(actual), len(actual) + len(forecast)))
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast,
                name='Future Forecast',
                line=dict(color='red', dash='dash'),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f'LSTM Forecast for {stored_column}',
                xaxis_title='Time Step',
                yaxis_title='Value',
                template=PLOTLY_TEMPLATE,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display future values
            st.subheader("🔮 Future Predictions")
            forecast_df = pd.DataFrame({
                'Step': range(1, len(forecast) + 1),
                'Predicted Value': forecast
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            st.info("✅ Model trained! Now upload a prediction file below to forecast from new data.")
        
        # Prediction on new data
        st.markdown("---")
        st.subheader("🔮 Forecast from New Data")
        
        stored_column = st.session_state.get('trained_lstm_column', ts_column)
        stored_seq_length = st.session_state.get('trained_lstm_seq_length', sequence_length)
        stored_forecast_horizon = st.session_state.get('trained_lstm_forecast_horizon', forecast_horizon)
        
        st.caption(f"Upload a CSV with column '{stored_column}' (at least {stored_seq_length} values needed)")
        
        lstm_predict_file = st.file_uploader("Upload New Time Series File (CSV)", type=['csv'], key="lstm_predict_file")
        
        if lstm_predict_file is not None:
            try:
                lstm_predict_df = pd.read_csv(lstm_predict_file)
                st.write(f"📊 Loaded {len(lstm_predict_df)} samples with {len(lstm_predict_df.columns)} columns")
                st.dataframe(lstm_predict_df.head(), use_container_width=True)
                
                if 'lstm_model' not in st.session_state:
                    st.warning("⚠️ Train an LSTM model first before making predictions")
                else:
                    if stored_column not in lstm_predict_df.columns:
                        st.error(f"❌ Column '{stored_column}' not found in prediction file")
                        st.info(f"Available columns: {list(lstm_predict_df.columns)}")
                    else:
                        if st.button("🎯 Generate Forecasts", type="primary", key="lstm_predict_btn"):
                            with st.spinner("Generating forecasts..."):
                                try:
                                    lstm_model = st.session_state['lstm_model']
                                    pred_results = lstm_model.lstm_predict(
                                        new_data=lstm_predict_df,
                                        column=stored_column
                                    )
                                    
                                    if 'error' in pred_results:
                                        st.error(f"❌ {pred_results['error']}")
                                    else:
                                        st.success(f"✅ Forecast complete!")
                                        
                                        # Show future forecast
                                        st.subheader("🔮 Future Forecast")
                                        future_forecast = pred_results['future_forecast']
                                        
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Forecast Steps", len(future_forecast))
                                        col2.metric("Min Forecast", f"{np.min(future_forecast):.4f}")
                                        col3.metric("Max Forecast", f"{np.max(future_forecast):.4f}")
                                        
                                        # Plot
                                        fig = go.Figure()
                                        
                                        # Original data
                                        original = lstm_predict_df[stored_column].values
                                        fig.add_trace(go.Scatter(
                                            y=original,
                                            name='Input Data',
                                            line=dict(color='blue')
                                        ))
                                        
                                        # Future forecast
                                        forecast_x = list(range(len(original), len(original) + len(future_forecast)))
                                        fig.add_trace(go.Scatter(
                                            x=forecast_x,
                                            y=future_forecast,
                                            name='Forecast',
                                            line=dict(color='red', dash='dash'),
                                            mode='lines+markers'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f'LSTM Forecast from New Data: {stored_column}',
                                            xaxis_title='Time Step',
                                            yaxis_title='Value',
                                            template=PLOTLY_TEMPLATE,
                                            height=500
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Forecast table
                                        forecast_df = pd.DataFrame({
                                            'Step': range(1, len(future_forecast) + 1),
                                            'Predicted Value': future_forecast
                                        })
                                        st.dataframe(forecast_df, use_container_width=True)
                                        
                                        # Download button
                                        csv = forecast_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Forecast",
                                            data=csv,
                                            file_name="lstm_forecast.csv",
                                            mime="text/csv"
                                        )
                                        
                                except Exception as e:
                                    st.error(f"❌ Forecast failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to load prediction file: {str(e)}")
    
    # =========================================================================
    # Autoencoder
    # =========================================================================
    else:  # Autoencoder
        st.subheader("🚨 Autoencoder Anomaly Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_cols = st.multiselect(
                "Features for Anomaly Detection",
                numeric_cols,
                default=numeric_cols[:5],
                key="ae_features"
            )
            encoding_dim = st.slider("Encoding Dimension", 2, 32, 8, key="ae_encoding")
        
        with col2:
            contamination = st.slider("Expected Anomaly Rate", 0.01, 0.20, 0.05, key="ae_contamination")
            epochs = st.slider("Epochs", 10, 200, 50, key="ae_epochs")
        
        with st.expander("🔧 Autoencoder Architecture", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers = st.multiselect(
                    "Hidden Layer Sizes",
                    [16, 32, 64, 128, 256],
                    default=[64, 32],
                    key="ae_hidden"
                )
            with col2:
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1, key="ae_dropout")
                batch_size = st.slider("Batch Size", 16, 128, 32, key="ae_batch")
        
        if st.button("🚀 Train Autoencoder", type="primary", key="train_ae"):
            if len(feature_cols) < 2:
                st.error("Select at least 2 features")
                return
            
            with st.spinner("Training Autoencoder..."):
                try:
                    results = nn.autoencoder_anomaly_detection(
                        features=feature_cols,
                        encoding_dim=encoding_dim,
                        hidden_layers=hidden_layers if hidden_layers else [64, 32],
                        dropout_rate=dropout,
                        epochs=epochs,
                        batch_size=batch_size,
                        contamination=contamination,
                        verbose=0
                    )
                    
                    # Store trained model info in session state for prediction
                    st.session_state['ae_model'] = nn
                    st.session_state['trained_ae_features'] = feature_cols
                    st.session_state['trained_ae_threshold'] = results['threshold']
                    st.session_state['ae_training_results'] = results
                    st.session_state['ae_trained'] = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Autoencoder Training failed: {str(e)}")
        
        # Display training results if model was trained (persists after file upload)
        if st.session_state.get('ae_trained', False) and 'ae_training_results' in st.session_state:
            results = st.session_state['ae_training_results']
            stored_ae_features_display = st.session_state.get('trained_ae_features', feature_cols)
            
            st.success("✅ Autoencoder Training Complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomalies Found", results['n_anomalies'])
            col2.metric("Anomaly Rate", f"{results['anomaly_percentage']:.2f}%")
            col3.metric("Threshold", f"{results['threshold']:.6f}")
            
            # Reconstruction error plot
            st.subheader("📊 Reconstruction Error Analysis")
            
            fig = go.Figure()
            
            reconstruction_errors = results['reconstruction_errors']
            threshold = results['threshold']
            anomaly_idx = results['anomaly_indices']
            
            # All points
            fig.add_trace(go.Scatter(
                y=reconstruction_errors,
                mode='lines',
                name='Reconstruction Error',
                line=dict(color='blue', width=1)
            ))
            
            # Threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold:.4f})"
            )
            
            # Highlight anomalies
            if len(anomaly_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_idx,
                    y=[reconstruction_errors[i] for i in anomaly_idx],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title='Autoencoder Anomaly Detection',
                xaxis_title='Sample Index',
                yaxis_title='Reconstruction Error',
                template=PLOTLY_TEMPLATE,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            if len(anomaly_idx) > 0:
                st.subheader("🔍 Anomaly Details")
                anomaly_df = df.iloc[anomaly_idx][stored_ae_features_display].copy()
                anomaly_df['Reconstruction Error'] = [reconstruction_errors[i] for i in anomaly_idx]
                anomaly_df = anomaly_df.sort_values('Reconstruction Error', ascending=False)
                st.dataframe(anomaly_df.head(20), use_container_width=True)
            
            st.info("✅ Model trained! Now upload a prediction file below to detect anomalies in new data.")
        
        # Prediction on new data
        st.markdown("---")
        st.subheader("🔮 Detect Anomalies in New Data")
        
        # Check if we have a properly trained model
        has_trained_ae = 'ae_model' in st.session_state and 'trained_ae_features' in st.session_state
        
        if has_trained_ae:
            stored_ae_features = st.session_state['trained_ae_features']
            stored_threshold = st.session_state['trained_ae_threshold']
            
            st.caption(f"Upload a CSV with features ({', '.join(stored_ae_features[:3])}{'...' if len(stored_ae_features) > 3 else ''}) to detect anomalies")
            st.info(f"**Trained model expects:** Features: {stored_ae_features[:5]}{'...' if len(stored_ae_features) > 5 else ''}, Threshold: {stored_threshold:.6f}")
        else:
            st.info("💡 Train an autoencoder above first, then upload a prediction file here.")
            stored_ae_features = feature_cols
        
        ae_predict_file = st.file_uploader("Upload Prediction/Test File (CSV)", type=['csv'], key="ae_predict_file")
        
        if ae_predict_file is not None:
            try:
                ae_predict_df = pd.read_csv(ae_predict_file)
                st.write(f"📊 Loaded {len(ae_predict_df)} samples with {len(ae_predict_df.columns)} columns")
                st.dataframe(ae_predict_df.head(), use_container_width=True)
                
                # Check if we have a trained model
                if not has_trained_ae:
                    st.warning("⚠️ Train an autoencoder first before detecting anomalies")
                else:
                    # Check features exist in prediction file
                    missing_features = [f for f in stored_ae_features if f not in ae_predict_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Missing features in prediction file: {missing_features}")
                        st.info(f"Expected features (from training): {stored_ae_features}")
                    
                    if st.button("🎯 Detect Anomalies", type="primary", key="ae_predict_btn"):
                        if missing_features:
                            st.error("Cannot predict - missing features")
                        else:
                            with st.spinner("Detecting anomalies..."):
                                try:
                                    ae_model = st.session_state['ae_model']
                                    pred_results = ae_model.predict_anomaly(
                                        new_data=ae_predict_df,
                                        features=stored_ae_features
                                    )
                                    
                                    if 'error' in pred_results:
                                        st.error(f"❌ {pred_results['error']}")
                                    else:
                                        reconstruction_errors = pred_results['reconstruction_errors']
                                        is_anomaly = pred_results['is_anomaly']
                                        anomaly_indices = pred_results['anomaly_indices']
                                        threshold = pred_results['threshold']
                                        
                                        n_anomalies = len(anomaly_indices)
                                        anomaly_pct = (n_anomalies / len(ae_predict_df)) * 100
                                        
                                        st.success(f"✅ Anomaly detection complete for {len(ae_predict_df)} samples!")
                                        
                                        # Metrics
                                        st.subheader("📊 Anomaly Detection Results")
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("Total Samples", len(ae_predict_df))
                                        col2.metric("Anomalies Found", n_anomalies)
                                        col3.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
                                        col4.metric("Threshold", f"{threshold:.6f}")
                                        
                                        # Reconstruction error plot
                                        st.subheader("📈 Reconstruction Error Analysis")
                                        
                                        fig = go.Figure()
                                        
                                        # Normal points
                                        normal_idx = np.where(~is_anomaly)[0]
                                        fig.add_trace(go.Scatter(
                                            x=normal_idx.tolist(),
                                            y=[reconstruction_errors[i] for i in normal_idx],
                                            mode='markers',
                                            name='Normal',
                                            marker=dict(color='steelblue', size=6, opacity=0.6)
                                        ))
                                        
                                        # Anomaly points
                                        if len(anomaly_indices) > 0:
                                            fig.add_trace(go.Scatter(
                                                x=anomaly_indices,
                                                y=[reconstruction_errors[i] for i in anomaly_indices],
                                                mode='markers',
                                                name='Anomalies',
                                                marker=dict(color='red', size=10, symbol='x')
                                            ))
                                        
                                        # Threshold line
                                        fig.add_hline(
                                            y=threshold,
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Threshold ({threshold:.4f})"
                                        )
                                        
                                        fig.update_layout(
                                            title='Anomaly Detection on New Data',
                                            xaxis_title='Sample Index',
                                            yaxis_title='Reconstruction Error',
                                            template=PLOTLY_TEMPLATE,
                                            height=500
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Results table
                                        st.subheader("📋 Results Table")
                                        result_df = ae_predict_df.copy()
                                        result_df['Reconstruction_Error'] = reconstruction_errors
                                        result_df['Is_Anomaly'] = is_anomaly
                                        
                                        # Show anomalies first, sorted by reconstruction error
                                        result_df_sorted = result_df.sort_values(
                                            by=['Is_Anomaly', 'Reconstruction_Error'],
                                            ascending=[False, False]
                                        )
                                        st.dataframe(result_df_sorted, use_container_width=True)
                                        
                                        # Anomaly details
                                        if len(anomaly_indices) > 0:
                                            st.subheader("🔍 Anomaly Details")
                                            anomaly_detail_df = ae_predict_df.iloc[anomaly_indices][stored_ae_features].copy()
                                            anomaly_detail_df['Reconstruction Error'] = [reconstruction_errors[i] for i in anomaly_indices]
                                            anomaly_detail_df = anomaly_detail_df.sort_values('Reconstruction Error', ascending=False)
                                            st.dataframe(anomaly_detail_df, use_container_width=True)
                                        
                                        # Download button
                                        csv = result_df_sorted.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Results",
                                            data=csv,
                                            file_name="anomaly_detection_results.csv",
                                            mime="text/csv"
                                        )
                                        
                                except Exception as e:
                                    st.error(f"❌ Anomaly detection failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to load prediction file: {str(e)}")


