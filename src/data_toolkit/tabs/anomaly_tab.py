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

def render_anomaly_tab():
    """Render anomaly detection tab"""
    st.header("🚨 Anomaly Detection")
    st.caption("Detect outliers using Isolation Forest, LOF, MCD, One-Class SVM, DBSCAN, and Autoencoder")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Method",
            [
                "Isolation Forest",
                "Local Outlier Factor",
                "Minimum Covariance",
                "One-Class SVM",
                "DBSCAN",
                "Autoencoder"
            ]
        )
        if method in ("Isolation Forest", "Local Outlier Factor", "Minimum Covariance", "Autoencoder"):
            contamination = st.slider("Contamination (% anomalies)", 0.01, 0.5, 0.1)
        if method == "One-Class SVM":
            nu = st.slider("nu (fraction anomalies)", 0.01, 0.5, 0.05)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            gamma = st.selectbox("Gamma", ["scale", "auto"])
        if method == "DBSCAN":
            eps = st.slider("eps (neighborhood radius)", 0.05, 2.0, 0.5)
            min_samples = st.slider("min_samples", 2, 20, 5)

    with col2:
        if method == "Isolation Forest":
            n_estimators = st.slider("N Estimators", 50, 500, 100)
        if method == "Local Outlier Factor":
            n_neighbors = st.slider("n_neighbors", 5, 50, 20)
        if method == "Autoencoder":
            ae_epochs = st.slider("Epochs", 10, 200, 100)
            ae_encoding_dim = st.slider("Encoding Dim", 2, 32, 8)

    with col3:
        if st.button("🚨 Detect Anomalies", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                try:
                    if method == "Isolation Forest":
                        results = ml.isolation_forest_anomaly(features, contamination=contamination, n_estimators=n_estimators)
                    elif method == "Local Outlier Factor":
                        results = ml.local_outlier_factor(features, contamination=contamination, n_neighbors=n_neighbors)
                    elif method == "Minimum Covariance":
                        results = ml.minimum_covariance_determinant(features, contamination=contamination)
                    elif method == "One-Class SVM":
                        results = ml.one_class_svm_anomaly(features, nu=nu, kernel=kernel, gamma=gamma)
                    elif method == "DBSCAN":
                        results = ml.dbscan_anomaly(features, eps=eps, min_samples=min_samples)
                    elif method == "Autoencoder":
                        results = ml.autoencoder_anomaly(features, contamination=contamination, encoding_dim=ae_encoding_dim, epochs=ae_epochs)

                    st.session_state.analysis_results['anomaly'] = results
                    st.success("✅ Detection complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")


    if 'anomaly' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['anomaly']



        if 'error' not in results:

            anomaly_labels = results.get('anomaly_labels', [])
            n_anomalies = sum(1 for x in anomaly_labels if x == -1)
            n_normal = sum(1 for x in anomaly_labels if x == 1)
            pct_anomalies = (n_anomalies / len(anomaly_labels) * 100) if len(anomaly_labels) > 0 else 0

            st.subheader("📊 Anomaly Detection Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Anomalies Found", n_anomalies)
            col2.metric("🟢 Normal Points", n_normal)
            col3.metric("Anomaly Rate", f"{pct_anomalies:.1f}%")

            # Visualization
            if len(features) >= 2:
                # Use the same rows the model saw: drop rows with any NaN in selected features
                X = df[features].dropna().reset_index(drop=True)

                if X.shape[0] == 0:
                    st.warning("No complete rows available for plotting (NA values present).")
                else:
                    x_vals = X[features[0]].values
                    y_vals = X[features[1] if len(features) > 1 else features[0]].values

                    # Prefer predictions key from model results, fall back to anomaly_labels
                    preds = None
                    for key in ('predictions', 'anomaly_labels', 'preds'):
                        if key in results:
                            preds = results[key]
                            break

                    if preds is None or len(preds) != len(X):
                        labels = ['Unknown'] * len(X)
                    else:
                        try:
                            preds_list = list(preds)
                            labels = ['🔴 Anomaly' if int(p) == -1 else '🟢 Normal' for p in preds_list[:len(X)]]
                        except Exception:
                            labels = ['Unknown'] * len(X)

                    df_plot = pd.DataFrame({
                        features[0]: x_vals,
                        features[1]: y_vals,
                        'Status': labels
                    })

                    # Add original index for reference
                    df_plot['Original_Index'] = X.index

                    # Create figure with anomalies more prominent
                    fig = go.Figure()


                    # Always plot, even if no anomalies
                    normal_mask = df_plot['Status'] == '🟢 Normal'
                    anomaly_mask = df_plot['Status'] == '🔴 Anomaly'
                    # Plot normal points (even if all points are normal)
                    fig.add_trace(go.Scatter(
                        x=df_plot.loc[normal_mask, features[0]],
                        y=df_plot.loc[normal_mask, features[1]],
                        mode='markers',
                        name='Normal',
                        marker=dict(size=6, color='green', opacity=0.4),
                        hovertemplate=f'{features[0]}: %{{x:.3f}}<br>{features[1]}: %{{y:.3f}}<extra>Normal</extra>'
                    ))
                    # Plot anomalies (even if none)
                    fig.add_trace(go.Scatter(
                        x=df_plot.loc[anomaly_mask, features[0]],
                        y=df_plot.loc[anomaly_mask, features[1]],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(size=12, color='red', symbol='x', line=dict(width=2, color='darkred')),
                        hovertemplate=f'{features[0]}: %{{x:.3f}}<br>{features[1]}: %{{y:.3f}}<extra>⚠️ ANOMALY</extra>'
                    ))
                    fig.update_layout(
                        title=f'{method} Anomaly Detection Results',
                        xaxis_title=features[0],
                        yaxis_title=features[1],
                        template=PLOTLY_TEMPLATE,
                        height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show anomaly details table
                    st.subheader("🔍 Anomaly Details")

                    # Create full results dataframe
                    df_results = X.copy()
                    df_results['Status'] = labels
                    df_results['Anomaly_Score'] = results.get('anomaly_scores', [0] * len(X))

                    # Show anomalies table
                    df_anomalies = df_results[df_results['Status'] == '🔴 Anomaly'].copy()
                    if len(df_anomalies) > 0:
                        st.write(f"**Showing {len(df_anomalies)} anomalies:**")
                        st.dataframe(df_anomalies, use_container_width=True)
                    else:
                        st.info("No anomalies detected with current settings.")

                    # Export functionality
                    st.subheader("📥 Export Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Export all results
                        csv_all = df_results.to_csv(index=True)
                        st.download_button(
                            label="📥 Download All Results (CSV)",
                            data=csv_all,
                            file_name="anomaly_detection_all.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Export only anomalies
                        if len(df_anomalies) > 0:
                            csv_anomalies = df_anomalies.to_csv(index=True)
                            st.download_button(
                                label="📥 Download Anomalies Only (CSV)",
                                data=csv_anomalies,
                                file_name="anomaly_detection_anomalies.csv",
                                mime="text/csv"
                            )


