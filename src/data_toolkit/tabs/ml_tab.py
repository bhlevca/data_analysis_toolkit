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

from ml_models import MLModels

PLOTLY_TEMPLATE = "plotly_white"

def render_ml_tab():
    """Render Machine Learning tab with Regression and Classification models"""
    st.header("🤖 Machine Learning: Regression & Classification Models")
    st.caption("Train supervised learning models to predict values (regression) or categories (classification)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns in the sidebar.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    # Check for duplicate column selection (feature also selected as target)
    if target in features:
        st.error(f"⚠️ Target column '{target}' is also selected as a feature. Please remove it from features or choose a different target.")
        return

    # Initialize with dataframe
    ml = MLModels(df)

    # Track if classification data is valid (will be set below)
    classification_data_invalid = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Selection")

        # Choose between Regression and Classification
        task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

        if task_type == "Regression":
            model_type = st.selectbox(
                "Choose Regression Model",
                ["Linear Regression", "Ridge Regression", "Lasso Regression",
                 "ElasticNet", "Random Forest Regressor", "Gradient Boosting Regressor",
                 "Decision Tree Regressor", "K-Nearest Neighbors Regressor", "Support Vector Regressor (SVR)"],
                help="Regression models predict continuous numerical values"
            )
        else:
            model_type = st.selectbox(
                "Choose Classification Model",
                ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier",
                 "Decision Tree Classifier", "K-Nearest Neighbors (KNN)",
                 "Support Vector Machine (SVM)", "Naive Bayes (Gaussian)"],
                help="Classification models predict categorical labels/classes"
            )
            # Check if target is actually categorical (not continuous)
            unique_values = df[target].nunique()
            is_float_target = df[target].dtype in ['float64', 'float32']
            has_decimals = is_float_target and (df[target] % 1 != 0).any()

            if has_decimals:
                classification_data_invalid = True
                st.error(f"⚠️ Target '{target}' contains continuous (decimal) values. Classification requires discrete classes. Switch to Regression or use a target with discrete categories.")
            elif unique_values > 20:
                st.warning(f"⚠️ Target '{target}' has {unique_values} unique values. Classification works best with fewer classes (typically < 20).")

        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2)

        # Show data info
        st.info(f"📊 Training data: {len(df)} samples, {len(features)} features")

    with col2:
        st.subheader("Model Training")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            # Disable train button if classification data is invalid
            train_disabled = task_type == "Classification" and classification_data_invalid
            if st.button("🎯 Train Model", width='stretch', disabled=train_disabled):
                with st.spinner("Training..."):
                    results = ml.train_model(features, target, model_type, test_size=test_size)
                    st.session_state.analysis_results['ml_model'] = results
                    # Only persist model if training succeeded
                    if 'error' not in results:
                        st.session_state.trained_model = ml.model
                        st.session_state.trained_scaler = ml.scaler
                        st.session_state.trained_features = features
                        st.session_state.trained_target = target
                        st.session_state.trained_task_type = task_type
                        st.success("✅ Model trained!")
                    else:
                        st.error(f"Training failed: {results['error']}")

        with col_b:
            if st.button("🔄 Cross-Validation", width='stretch'):
                with st.spinner("Running CV..."):
                    # Correct API: cross_validation(features, target, cv, model_name)
                    cv_results = ml.cross_validation(features, target, cv=cv_folds, model_name=model_type)
                    st.session_state.analysis_results['cv_results'] = cv_results
                    st.success("Cross-validation complete!")

        with col_c:
            if st.button("📊 Feature Importance", width='stretch'):
                with st.spinner("Calculating..."):
                    # Correct API: feature_importance(features, target)
                    importance = ml.feature_importance(features, target)
                    st.session_state.analysis_results['feature_importance'] = importance

    st.markdown("---")

    # Display results
    if 'ml_model' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ml_model']

        if 'error' in results:
            st.error(results['error'])
        else:
            is_classifier = results.get('is_classifier', False)

            if is_classifier:
                # Classification results
                st.subheader("📈 Classification Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
                col2.metric("Precision", f"{results.get('precision', 0):.4f}")
                col3.metric("Recall", f"{results.get('recall', 0):.4f}")
                col4.metric("F1 Score", f"{results.get('f1_score', 0):.4f}")

                # Confusion Matrix
                if 'confusion_matrix' in results:
                    st.markdown("**Confusion Matrix:**")
                    cm = np.array(results['confusion_matrix'])
                    classes = results.get('classes', list(range(len(cm))))

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
                    st.plotly_chart(fig_cm, width='stretch')

                # Classification Report
                if 'classification_report' in results:
                    with st.expander("📋 Detailed Classification Report"):
                        st.text(results['classification_report'])
            else:
                # Regression results
                st.subheader("📈 Regression Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{results.get('r2', 0):.4f}",
                           help="Coefficient of determination: 1.0 = perfect fit, 0 = no predictive power")
                col2.metric("RMSE", f"{results.get('rmse', 0):.4f}",
                           help="Root Mean Square Error: average prediction error in target units")
                col3.metric("MSE", f"{results.get('mse', 0):.4f}",
                           help="Mean Squared Error: average of squared differences")
                # Calculate MAE if we have predictions
                if 'y_test' in results and 'predictions' in results:
                    mae = np.mean(np.abs(np.array(results['y_test']) - np.array(results['predictions'])))
                    col4.metric("MAE", f"{mae:.4f}",
                               help="Mean Absolute Error: average absolute difference")

                if 'coefficients' in results:
                    with st.expander("📋 Model Coefficients", expanded=False):
                        coef_df = pd.DataFrame({
                            'Feature': list(results['coefficients'].keys()),
                            'Coefficient': list(results['coefficients'].values())
                        })
                        st.dataframe(coef_df, width='stretch')
                        if 'intercept' in results:
                            st.write(f"**Intercept:** {results['intercept']:.4f}")

                # ═══════════════════════════════════════════════════════════════
                # PLOT 1: Training Data - Actual vs Model Predictions (Test Set)
                # ═══════════════════════════════════════════════════════════════
                st.markdown("### 📊 Plot 1: Model Evaluation on Test Set")

                if 'y_test' in results and 'predictions' in results:
                    try:
                        y_test = results['y_test']
                        y_pred = results['predictions']

                        # Convert to numpy arrays, handling pandas Series
                        if hasattr(y_test, 'values'):
                            y_test = y_test.values
                        y_test = np.array(y_test).flatten()
                        y_pred = np.array(y_pred).flatten()

                        st.caption(f"Showing {len(y_test)} test samples. Blue = Actual values, Orange = Model predictions.")

                        fig_train = go.Figure()

                        # Get X_test for x-axis if available
                        result_features = results.get('features', features)
                        if 'X_test' in results and len(result_features) > 0:
                            X_test = results['X_test']
                            x_feature = result_features[0]
                            if hasattr(X_test, 'values'):
                                x_vals = X_test[x_feature].values
                            else:
                                x_vals = np.array(X_test[x_feature])
                            x_label = x_feature

                            # Sort by x for cleaner visualization
                            sort_idx = np.argsort(x_vals)
                            x_sorted = x_vals[sort_idx]
                            y_test_sorted = y_test[sort_idx]
                            y_pred_sorted = y_pred[sort_idx]
                        else:
                            x_sorted = np.arange(len(y_test))
                            y_test_sorted = y_test
                            y_pred_sorted = y_pred
                            x_label = "Sample Index"

                        # Actual values (blue circles)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_test_sorted,
                            mode='markers',
                            name='Actual (Test Set)',
                            marker=dict(opacity=0.7, color='steelblue', size=10, symbol='circle')
                        ))

                        # Model predictions (orange diamonds)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_pred_sorted,
                            mode='markers',
                            name='Model Prediction',
                            marker=dict(opacity=0.8, color='darkorange', size=8, symbol='diamond')
                        ))

                        # Add trend line through predictions (sorted)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_pred_sorted,
                            mode='lines',
                            name='Prediction Trend',
                            line=dict(color='darkorange', width=2, dash='dash'),
                            showlegend=False
                        ))

                        fig_train.update_layout(
                            title=f'Model Evaluation: Actual vs Predicted ({target})',
                            xaxis_title=x_label,
                            yaxis_title=f'{target}',
                            template=PLOTLY_TEMPLATE,
                            height=500,
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                        )
                        st.plotly_chart(fig_train, width='stretch')

                        # Residual info
                        residuals = y_test_sorted - y_pred_sorted
                        with st.expander("📉 Residual Analysis"):
                            col_r1, col_r2, col_r3 = st.columns(3)
                            col_r1.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                            col_r2.metric("Std Residual", f"{np.std(residuals):.4f}")
                            col_r3.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.4f}")
                    except Exception as e:
                        st.error(f"Plot 1 error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.warning("No test data available for plotting. Train the model first.")

    # Predict on new data - ALWAYS show this section
    st.markdown("---")
    st.subheader("🔮 Predict on New Data")
    trained_model = st.session_state.get('trained_model')
    trained_features = st.session_state.get('trained_features', features)
    trained_scaler = st.session_state.get('trained_scaler')

    # Always show the upload option, even if no model is trained yet
    data_option = st.radio("Prediction data source", ["Use current data", "Upload new CSV"], index=0, key='ml_pred_source')
    new_df = df

    if data_option == "Upload new CSV":
        upload = st.file_uploader("Upload CSV for prediction", type=['csv'], key='ml_predict_upload')
        if upload is not None:
            try:
                new_df = pd.read_csv(upload)
                st.success(f"✅ Loaded {len(new_df)} rows for prediction")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                new_df = None

    if trained_model is None:
        st.info("💡 Train a model above first, then click Predict to make predictions on this data.")

    # Always show predict button (disabled state handled by logic)
    predict_disabled = trained_model is None
    if st.button("🔮 Predict", width='stretch', disabled=predict_disabled):
        if new_df is None:
            st.error("No data available for prediction")
        else:
            ml_pred = MLModels(new_df)
            ml_pred.model = trained_model
            ml_pred.scaler = trained_scaler
            pred_results = ml_pred.predict_new_data(new_df, trained_features)
            if 'error' in pred_results:
                st.error(pred_results['error'])
            else:
                st.session_state.analysis_results['ml_predictions'] = pred_results
                st.session_state.ml_prediction_df = new_df[trained_features].copy()
                # Save full dataframe to check for actual target values
                st.session_state.ml_prediction_full_df = new_df.copy()
                st.success(f"✅ Predicted {pred_results['n_rows']} rows")

    if 'ml_predictions' in st.session_state.analysis_results:
        preds = st.session_state.analysis_results['ml_predictions']
        if 'error' not in preds:
            # ═══════════════════════════════════════════════════════════════
            # PLOT 2: New Data Predictions
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📊 Plot 2: Predictions on New Data")

            base_df = st.session_state.get('ml_prediction_df')
            full_df = st.session_state.get('ml_prediction_full_df')  # Full df with target column
            plot_features = st.session_state.get('trained_features', [])
            trained_target = st.session_state.get('trained_target', 'target')
            trained_task_type = st.session_state.get('trained_task_type', 'Regression')
            is_classification = trained_task_type == "Classification"

            if base_df is not None:
                try:
                    preview_df = base_df.iloc[:len(preds['predictions'])].copy()
                    preview_df['prediction'] = preds['predictions']

                    # Check if new data has actual target values for comparison
                    has_actual = False
                    if full_df is not None and trained_target in full_df.columns:
                        has_actual = True
                        actual_values = full_df[trained_target].iloc[:len(preds['predictions'])].values
                        preview_df['actual'] = actual_values

                    # Prediction metrics (if actual values available)
                    if has_actual:
                        pred_vals = np.array(preds['predictions'])
                        actual_vals = np.array(actual_values)

                        if is_classification:
                            # ═══════════════════════════════════════════════════════════════
                            # CLASSIFICATION METRICS
                            # ═══════════════════════════════════════════════════════════════
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix
                            try:
                                # Convert predictions to same type as actuals if needed
                                if actual_vals.dtype != pred_vals.dtype:
                                    pred_vals = pred_vals.astype(actual_vals.dtype)
                                
                                accuracy = accuracy_score(actual_vals, pred_vals)
                                # Use average='weighted' for multi-class
                                precision = precision_score(actual_vals, pred_vals, average='weighted', zero_division=0)
                                recall = recall_score(actual_vals, pred_vals, average='weighted', zero_division=0)
                                f1 = f1_score(actual_vals, pred_vals, average='weighted', zero_division=0)
                                
                                st.markdown("**📈 Classification Metrics (comparing to actual labels in new data):**")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy:.4f}",
                                           help="Proportion of correct predictions")
                                col2.metric("Precision", f"{precision:.4f}",
                                           help="Weighted precision across classes")
                                col3.metric("Recall", f"{recall:.4f}",
                                           help="Weighted recall across classes")
                                col4.metric("F1 Score", f"{f1:.4f}",
                                           help="Weighted F1 score")
                                
                                # Confusion Matrix
                                cm = sk_confusion_matrix(actual_vals, pred_vals)
                                classes = np.unique(np.concatenate([actual_vals, pred_vals]))
                                
                                st.markdown("**🔲 Confusion Matrix (Predictions vs Actual):**")
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
                                    title='Confusion Matrix: New Data Predictions',
                                    xaxis_title='Predicted',
                                    yaxis_title='Actual',
                                    template=PLOTLY_TEMPLATE,
                                    height=400
                                )
                                st.plotly_chart(fig_cm, width='stretch')
                                
                            except Exception as e:
                                st.error(f"Error computing classification metrics: {e}")
                        else:
                            # ═══════════════════════════════════════════════════════════════
                            # REGRESSION METRICS
                            # ═══════════════════════════════════════════════════════════════
                            # Filter out NaN values
                            mask = ~(np.isnan(pred_vals.astype(float)) | np.isnan(actual_vals.astype(float)))
                            if mask.sum() > 0:
                                pred_clean = pred_vals[mask].astype(float)
                                actual_clean = actual_vals[mask].astype(float)

                                pred_r2 = 1 - np.sum((actual_clean - pred_clean)**2) / np.sum((actual_clean - np.mean(actual_clean))**2)
                                pred_rmse = np.sqrt(np.mean((actual_clean - pred_clean)**2))
                                pred_mae = np.mean(np.abs(actual_clean - pred_clean))

                                st.markdown("**📈 Regression Metrics (comparing to actual values in new data):**")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("R² Score", f"{pred_r2:.4f}",
                                           help="How well predictions match actuals: 1.0 = perfect")
                                col2.metric("RMSE", f"{pred_rmse:.4f}",
                                           help="Root Mean Square Error")
                                col3.metric("MAE", f"{pred_mae:.4f}",
                                           help="Mean Absolute Error")
                                col4.metric("N Samples", f"{len(pred_clean)}",
                                           help="Number of samples compared")

                    # Summary stats and visualization - handle classification vs regression differently
                    if is_classification:
                        # ═══════════════════════════════════════════════════════════════
                        # CLASSIFICATION VISUALIZATION
                        # ═══════════════════════════════════════════════════════════════
                        st.markdown("**📊 Prediction Distribution by Class:**")
                        
                        pred_counts = pd.Series(preds['predictions']).value_counts().sort_index()
                        
                        if has_actual:
                            actual_counts = pd.Series(actual_values).value_counts().sort_index()
                            all_classes = sorted(set(pred_counts.index) | set(actual_counts.index))
                            
                            # Create grouped bar chart
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Bar(
                                name='Actual',
                                x=[str(c) for c in all_classes],
                                y=[actual_counts.get(c, 0) for c in all_classes],
                                marker_color='steelblue'
                            ))
                            fig_dist.add_trace(go.Bar(
                                name='Predicted',
                                x=[str(c) for c in all_classes],
                                y=[pred_counts.get(c, 0) for c in all_classes],
                                marker_color='crimson'
                            ))
                            fig_dist.update_layout(
                                title='Class Distribution: Actual vs Predicted',
                                xaxis_title='Class',
                                yaxis_title='Count',
                                barmode='group',
                                template=PLOTLY_TEMPLATE,
                                height=400
                            )
                        else:
                            fig_dist = go.Figure(data=[
                                go.Bar(
                                    x=[str(c) for c in pred_counts.index],
                                    y=pred_counts.values,
                                    marker_color='crimson'
                                )
                            ])
                            fig_dist.update_layout(
                                title='Predicted Class Distribution',
                                xaxis_title='Class',
                                yaxis_title='Count',
                                template=PLOTLY_TEMPLATE,
                                height=400
                            )
                        st.plotly_chart(fig_dist, width='stretch')
                        
                        # Summary stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("N Predictions", len(preds['predictions']))
                        col2.metric("Unique Classes", len(pred_counts))
                        col3.metric("Most Common", str(pred_counts.idxmax()))
                        col4.metric("Most Common Count", int(pred_counts.max()))
                    else:
                        # ═══════════════════════════════════════════════════════════════
                        # REGRESSION VISUALIZATION
                        # ═══════════════════════════════════════════════════════════════
                        # Summary stats for predictions
                        st.markdown("**📊 Prediction Summary:**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Min Prediction", f"{np.min(preds['predictions']):.4f}")
                        col2.metric("Max Prediction", f"{np.max(preds['predictions']):.4f}")
                        col3.metric("Mean Prediction", f"{np.mean(preds['predictions']):.4f}")
                        col4.metric("Std Prediction", f"{np.std(preds['predictions']):.4f}")

                        # Create visualization (regression only - scatter plot)
                        if len(plot_features) >= 1:
                            x_feature = plot_features[0]
                            has_x_feature = x_feature in preview_df.columns

                            if has_x_feature:
                                # Sort by x for cleaner visualization
                                sort_idx = np.argsort(preview_df[x_feature].values)
                                x_sorted = preview_df[x_feature].values[sort_idx]
                                pred_sorted = preview_df['prediction'].values[sort_idx]

                                fig_pred = go.Figure()

                                # If we have actual values, show them
                                if has_actual:
                                    actual_sorted = preview_df['actual'].values[sort_idx]
                                    st.caption("🔵 Blue = Actual values (from new data) | 🔴 Red = Model predictions")

                                    fig_pred.add_trace(go.Scatter(
                                        x=x_sorted,
                                        y=actual_sorted,
                                        mode='markers',
                                        name=f'Actual ({trained_target})',
                                        marker=dict(color='steelblue', size=10, opacity=0.7, symbol='circle')
                                    ))
                                else:
                                    st.caption("🔴 Red diamonds = Model predictions for new data inputs")

                                # Predictions
                                fig_pred.add_trace(go.Scatter(
                                    x=x_sorted,
                                    y=pred_sorted,
                                    mode='markers',
                                    name='Predictions',
                                    marker=dict(color='crimson', size=10, opacity=0.9, symbol='diamond')
                                ))

                                # Prediction trend line
                                fig_pred.add_trace(go.Scatter(
                                    x=x_sorted,
                                    y=pred_sorted,
                                    mode='lines',
                                    name='Prediction Trend',
                                    line=dict(color='crimson', width=2, dash='dash'),
                                    showlegend=False
                                ))

                                fig_pred.update_layout(
                                    title=f'New Data: Predictions vs {x_feature}',
                                    xaxis_title=x_feature,
                                    yaxis_title=trained_target,
                                    template=PLOTLY_TEMPLATE,
                                    height=500,
                                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                                )
                                st.plotly_chart(fig_pred, width='stretch')
                            else:
                                # Fallback: plot by index
                                st.caption("🔴 Red diamonds = Model predictions")
                                fig_pred = go.Figure()
                                fig_pred.add_trace(go.Scatter(
                                    x=list(range(len(preview_df))),
                                    y=preview_df['prediction'].values,
                                    mode='markers+lines',
                                    name='Predictions',
                                    marker=dict(color='crimson', size=10, symbol='diamond'),
                                    line=dict(color='crimson', width=1, dash='dot')
                                ))
                                fig_pred.update_layout(
                                    title='Predictions by Sample Index',
                                    xaxis_title='Sample Index',
                                    yaxis_title='Predicted Value',
                                    template=PLOTLY_TEMPLATE,
                                    height=450
                                )
                                st.plotly_chart(fig_pred, width='stretch')

                    # Data table
                    with st.expander("📋 Predictions Table (first 100 rows)", expanded=False):
                        st.dataframe(preview_df.head(100), width='stretch')

                        # Download button
                        csv = preview_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.dataframe(pd.DataFrame({'prediction': preds['predictions']}).head(100), width='stretch')
            else:
                st.dataframe(pd.DataFrame({'prediction': preds['predictions']}).head(100), width='stretch')

    if 'cv_results' in st.session_state.analysis_results:
        cv = st.session_state.analysis_results['cv_results']
        st.subheader("🔄 Cross-Validation Results")
        st.write(f"Mean R²: {cv['mean']:.4f} ± {cv['std']:.4f}")

        fig = go.Figure(data=[
            go.Bar(x=[f'Fold {i+1}' for i in range(len(cv['scores']))],
                  y=cv['scores'], marker_color='steelblue')
        ])
        fig.add_hline(y=cv['mean'], line_dash='dash', line_color='red')
        fig.update_layout(title='CV Scores by Fold', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')

    if 'feature_importance' in st.session_state.analysis_results:
        importance = st.session_state.analysis_results['feature_importance']
        st.subheader("📊 Feature Importance")

        fig = go.Figure(data=[
            go.Bar(y=list(importance.keys()), x=list(importance.values()),
                  orientation='h', marker_color='steelblue')
        ])
        fig.update_layout(title='Feature Importance', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')


