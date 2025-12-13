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

from visualization_methods import VisualizationMethods

PLOTLY_TEMPLATE = "plotly_white"

def render_visualization_tab():
    """Render visualization tab with interactive charts and regression lines"""
    st.header("📈 Visualization & Plots")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    col1, col2 = st.columns([1, 2])

    with col1:
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Scatter Matrix", "Correlation Heatmap", "Box Plots",
             "Distribution Plots", "3D Scatter", "Parallel Coordinates",
             "Linear Regression Plot (with Statistics)"]
        )

        if plot_type == "Correlation Heatmap":
            corr_method = st.selectbox("Method", ['pearson', 'spearman', 'kendall'])

        if plot_type == "3D Scatter" and len(features) >= 3:
            x_3d = st.selectbox("X axis", features, index=0)
            y_3d = st.selectbox("Y axis", features, index=min(1, len(features)-1))
            z_3d = st.selectbox("Z axis", features, index=min(2, len(features)-1))

        if plot_type == "Linear Regression Plot (with Statistics)":
            st.markdown("**Select variables for regression:**")
            x_reg = st.selectbox("X variable (independent)", features, index=0, key="reg_x")
            available_y = [f for f in features if f != x_reg]
            if st.session_state.target_col and st.session_state.target_col not in available_y:
                available_y.append(st.session_state.target_col)
            if available_y:
                y_reg = st.selectbox("Y variable (dependent)", available_y, index=0, key="reg_y")
            else:
                y_reg = st.selectbox("Y variable (dependent)", features, index=min(1, len(features)-1), key="reg_y")
            show_ci = st.checkbox("Show 95% Confidence Interval", value=True, key="reg_ci")

    if st.button("📊 Generate Plot", use_container_width=True):
        with st.spinner("Creating visualization..."):

            if plot_type == "Scatter Matrix":
                fig = px.scatter_matrix(
                    df[features[:5]],
                    title="Scatter Matrix (Interactive!)",
                    template=PLOTLY_TEMPLATE
                )
                fig.update_traces(diagonal_visible=True)
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Correlation Heatmap":
                corr = df[features].corr(method=corr_method)
                fig = px.imshow(
                    corr,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f'{corr_method.capitalize()} Correlation Heatmap'
                )
                fig.update_layout(height=600, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Box Plots":
                box_data = df[features].melt(var_name='Feature', value_name='Value')
                fig = px.box(box_data, x='Feature', y='Value',
                           title='Box Plots', template=PLOTLY_TEMPLATE, points='outliers')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Distribution Plots":
                for col in features[:4]:
                    fig = px.histogram(df, x=col, marginal='box',
                                      title=f'Distribution of {col}', template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "3D Scatter":
                if len(features) >= 3:
                    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                                       title=f'3D Scatter', template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 3 features")

            elif plot_type == "Parallel Coordinates":
                fig = px.parallel_coordinates(
                    df[features],
                    title='Parallel Coordinates',
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Linear Regression Plot (with Statistics)":
                # Import scipy for regression statistics
                from scipy import stats as scipy_stats

                # Get data
                x_data = df[x_reg].dropna()
                y_data = df[y_reg].dropna()

                # Align indices
                common_idx = x_data.index.intersection(y_data.index)
                x_data = x_data.loc[common_idx].values
                y_data = y_data.loc[common_idx].values

                if len(x_data) < 3:
                    st.error("Need at least 3 data points for regression.")
                else:
                    # Perform linear regression using scipy.stats.linregress
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
                    r_squared = r_value ** 2

                    # Create regression line
                    x_line = np.array([x_data.min(), x_data.max()])
                    y_line = slope * x_line + intercept

                    # Create figure
                    fig = go.Figure()

                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_data,
                        mode='markers',
                        name='Data Points',
                        marker=dict(color='steelblue', size=8, opacity=0.7)
                    ))

                    # Regression line
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        name=f'Regression Line',
                        line=dict(color='red', width=2)
                    ))

                    # Add confidence interval if requested
                    if show_ci:
                        n = len(x_data)
                        x_mean = np.mean(x_data)
                        ss_x = np.sum((x_data - x_mean) ** 2)
                        y_pred = slope * x_data + intercept
                        residuals = y_data - y_pred
                        mse = np.sum(residuals ** 2) / (n - 2)
                        se = np.sqrt(mse)

                        # For confidence interval
                        t_val = scipy_stats.t.ppf(0.975, n - 2)

                        # Calculate CI at many points for smooth band
                        x_ci = np.linspace(x_data.min(), x_data.max(), 100)
                        y_ci = slope * x_ci + intercept
                        se_fit = se * np.sqrt(1/n + (x_ci - x_mean)**2 / ss_x)
                        ci_upper = y_ci + t_val * se_fit
                        ci_lower = y_ci - t_val * se_fit

                        # Add CI band
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([x_ci, x_ci[::-1]]),
                            y=np.concatenate([ci_upper, ci_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                            name='95% Confidence Interval',
                            hoverinfo='skip'
                        ))

                    # Format equation and stats
                    sign = '+' if intercept >= 0 else '-'
                    eq_str = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"

                    # Update layout with stats annotation
                    fig.update_layout(
                        title=dict(
                            text=f'Linear Regression: {y_reg} vs {x_reg}<br>' +
                                 f'<span style="font-size:14px">{eq_str} | R² = {r_squared:.4f} | p = {p_value:.2e}</span>',
                            font=dict(size=16)
                        ),
                        xaxis_title=f'{x_reg} (Independent Variable)',
                        yaxis_title=f'{y_reg} (Dependent Variable)',
                        template=PLOTLY_TEMPLATE,
                        height=550,
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics in a clear table
                    st.markdown("### 📊 Regression Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Slope (m)", f"{slope:.6f}")
                    col2.metric("Intercept (b)", f"{intercept:.6f}")
                    col3.metric("R² (R-squared)", f"{r_squared:.4f}")
                    col4.metric("p-value", f"{p_value:.2e}")

                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Correlation (r)", f"{r_value:.4f}")
                    col6.metric("Std Error (slope)", f"{std_err:.6f}")
                    col7.metric("N (samples)", f"{len(x_data)}")
                    col8.metric("Degrees of Freedom", f"{len(x_data) - 2}")

                    # Interpretation
                    st.markdown("### 📖 Interpretation")

                    interpretation = []

                    # Slope interpretation
                    if slope > 0:
                        interpretation.append(f"- **Positive relationship**: For each 1-unit increase in {x_reg}, {y_reg} increases by {slope:.4f} units.")
                    else:
                        interpretation.append(f"- **Negative relationship**: For each 1-unit increase in {x_reg}, {y_reg} decreases by {abs(slope):.4f} units.")

                    # R² interpretation
                    if r_squared >= 0.9:
                        interpretation.append(f"- **Excellent fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance in {y_reg} is explained by {x_reg}.")
                    elif r_squared >= 0.7:
                        interpretation.append(f"- **Good fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance is explained.")
                    elif r_squared >= 0.5:
                        interpretation.append(f"- **Moderate fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance is explained.")
                    else:
                        interpretation.append(f"- **Weak fit**: R² = {r_squared:.4f} means only {r_squared*100:.1f}% of variance is explained. Consider non-linear models.")

                    # p-value interpretation
                    if p_value < 0.001:
                        interpretation.append(f"- **Highly significant**: p = {p_value:.2e} (p < 0.001). Very strong evidence of a relationship.")
                    elif p_value < 0.01:
                        interpretation.append(f"- **Very significant**: p = {p_value:.4f} (p < 0.01). Strong evidence of a relationship.")
                    elif p_value < 0.05:
                        interpretation.append(f"- **Significant**: p = {p_value:.4f} (p < 0.05). Moderate evidence of a relationship.")
                    else:
                        interpretation.append(f"- **Not significant**: p = {p_value:.4f} (p ≥ 0.05). Insufficient evidence of a linear relationship.")

                    for line in interpretation:
                        st.markdown(line)


