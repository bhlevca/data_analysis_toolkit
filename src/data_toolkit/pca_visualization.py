"""
PCA Vector Visualization Enhancement
Adds feature vectors and interpretation to PCA scatter plots
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple


def create_pca_biplot_with_vectors(
    transformed_data: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    feature_names: list,
    color_by: np.ndarray = None,
    scale_factor: float = 1.0
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create enhanced PCA biplot with feature vectors and interpretation
    
    Parameters:
    -----------
    transformed_data : np.ndarray
        Transformed data in PC space (n_samples x n_components)
    components : np.ndarray
        Loading matrix from PCA (n_features x n_components)
    explained_variance : np.ndarray
        Explained variance ratio for each component
    feature_names : list
        Names of original features
    color_by : np.ndarray, optional
        Array to color points by (e.g., class labels)
    scale_factor : float
        Scale vectors for visibility (default 1.0)
    
    Returns:
    --------
    fig : go.Figure
        Plotly figure with biplot
    vector_interpretation : Dict
        Interpretation of vectors and relationships
    """
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot of transformed data
    if color_by is not None:
        unique_classes = np.unique(color_by)
        colors = px.colors.qualitative.Plotly
        
        for i, cls in enumerate(unique_classes):
            mask = color_by == cls
            fig.add_trace(go.Scatter(
                x=transformed_data[mask, 0],
                y=transformed_data[mask, 1],
                mode='markers',
                name=f'Class {cls}' if isinstance(cls, (int, float)) else str(cls),
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                )
            ))
    else:
        fig.add_trace(go.Scatter(
            x=transformed_data[:, 0],
            y=transformed_data[:, 1],
            mode='markers',
            name='Data Points',
            marker=dict(
                size=6,
                color='steelblue',
                opacity=0.6,
                line=dict(width=0.5, color='white')
            )
        ))
    
    # Add feature vectors
    vector_info = {}
    
    for i, feature_name in enumerate(feature_names):
        # Scale vectors for visibility
        x_vec = components[i, 0] * scale_factor
        y_vec = components[i, 1] * scale_factor
        
        # Calculate vector magnitude (importance)
        magnitude = np.sqrt(x_vec**2 + y_vec**2)
        
        # Store vector information
        vector_info[feature_name] = {
            'x': components[i, 0],
            'y': components[i, 1],
            'magnitude': magnitude,
            'angle_rad': np.arctan2(y_vec, x_vec),
            'angle_deg': np.degrees(np.arctan2(y_vec, x_vec))
        }
        
        # Add arrow from origin to vector endpoint using line with arrowhead annotation
        fig.add_trace(go.Scatter(
            x=[0, x_vec],
            y=[0, y_vec],
            mode='lines',
            name=feature_name,
            line=dict(width=2),
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add arrowhead at end of vector using annotation
        fig.add_annotation(
            x=x_vec,
            y=y_vec,
            ax=0,
            ay=0,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        )
        
        # Add label at vector endpoint
        fig.add_annotation(
            x=x_vec * 1.1,
            y=y_vec * 1.1,
            text=feature_name,
            showarrow=False,
            font=dict(size=11, color='darkblue'),
            xanchor='center',
            yanchor='middle'
        )
    
    # Calculate and display PC variance info
    pc1_var = explained_variance[0] * 100 if len(explained_variance) > 0 else 0
    pc2_var = explained_variance[1] * 100 if len(explained_variance) > 1 else 0
    
    # Update layout
    fig.update_layout(
        title=f'PCA Biplot with Feature Vectors<br><sub>PC1: {pc1_var:.1f}% | PC2: {pc2_var:.1f}%</sub>',
        xaxis_title=f'PC1 ({pc1_var:.1f}% variance)',
        yaxis_title=f'PC2 ({pc2_var:.1f}% variance)',
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        height=700,
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'),
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    # Add equal aspect ratio hint
    max_range = max(
        np.max(np.abs(transformed_data[:, 0])),
        np.max(np.abs(transformed_data[:, 1]))
    )
    fig.update_xaxes(range=[-max_range*1.2, max_range*1.2])
    fig.update_yaxes(range=[-max_range*1.2, max_range*1.2])
    
    return fig, vector_info


def interpret_vectors(
    vector_info: Dict[str, Any],
    feature_names: list
) -> Dict[str, str]:
    """
    Generate interpretation text for PCA vectors
    
    Returns interpretation of:
    - Which features drive each PC
    - Feature correlations (based on vector angles)
    - Feature importance (based on vector magnitude)
    """
    
    interpretation = {}
    
    # PC1 drivers (largest x-component magnitude)
    pc1_drivers = sorted(
        vector_info.items(),
        key=lambda x: abs(x[1]['x']),
        reverse=True
    )[:3]
    
    # PC2 drivers (largest y-component magnitude)
    pc2_drivers = sorted(
        vector_info.items(),
        key=lambda x: abs(x[1]['y']),
        reverse=True
    )[:3]
    
    # Format interpretations
    pc1_text = "PC1 is primarily driven by: " + ", ".join([f[0] for f in pc1_drivers])
    pc2_text = "PC2 is primarily driven by: " + ", ".join([f[0] for f in pc2_drivers])
    
    interpretation['pc1_drivers'] = pc1_text
    interpretation['pc2_drivers'] = pc2_text
    
    # Feature correlations (based on vector angles)
    correlations = []
    features = list(vector_info.keys())
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            feat1, feat2 = features[i], features[j]
            angle1 = vector_info[feat1]['angle_rad']
            angle2 = vector_info[feat2]['angle_rad']
            
            angle_diff = abs(angle1 - angle2)
            # Normalize to 0-180 degrees
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            angle_deg = np.degrees(angle_diff)
            
            if angle_deg < 30:
                relationship = f"{feat1} and {feat2}: Strongly correlated (parallel vectors)"
            elif angle_deg < 60:
                relationship = f"{feat1} and {feat2}: Moderately correlated"
            elif angle_deg < 120:
                relationship = f"{feat1} and {feat2}: Weakly related/Independent"
            elif angle_deg < 150:
                relationship = f"{feat1} and {feat2}: Negatively correlated"
            else:
                relationship = f"{feat1} and {feat2}: Strongly negatively correlated (opposite)"
            
            correlations.append(relationship)
    
    interpretation['correlations'] = correlations
    
    # Feature importance
    importances = sorted(
        vector_info.items(),
        key=lambda x: x[1]['magnitude'],
        reverse=True
    )
    
    importance_text = "Feature importance (by vector magnitude):\n"
    for feat, info in importances:
        importance_text += f"  {feat}: {info['magnitude']:.3f}\n"
    
    interpretation['feature_importance'] = importance_text
    
    return interpretation


def generate_pca_insights(
    vector_info: Dict[str, Any],
    explained_variance: np.ndarray,
    total_variance: float
) -> str:
    """
    Generate human-readable insights from PCA analysis
    """
    
    insights = []
    
    # Variance insight
    pc1_var = explained_variance[0] * 100 if len(explained_variance) > 0 else 0
    pc2_var = explained_variance[1] * 100 if len(explained_variance) > 1 else 0
    total = (pc1_var + pc2_var)
    
    insights.append(f"## PCA Analysis Summary\n")
    insights.append(f"- **PC1** explains {pc1_var:.1f}% of variance")
    insights.append(f"- **PC2** explains {pc2_var:.1f}% of variance")
    insights.append(f"- **Total** (PC1 + PC2): {total:.1f}% of total variance")
    
    if total > 90:
        insights.append(f"- ✅ Excellent! 2 components capture {total:.1f}% of information")
    elif total > 80:
        insights.append(f"- ✅ Good! 2 components capture {total:.1f}% of information")
    elif total > 70:
        insights.append(f"- ⚠️ Acceptable. Consider using 3+ components for {total:.1f}%")
    else:
        insights.append(f"- ❌ Low coverage. Use 3+ components to capture more variance")
    
    return "\n".join(insights)
