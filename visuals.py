"""
visuals.py
Visualization helpers for the Student Performance Predictor dashboard.
Uses Plotly so charts match the clean, modern dashboard look.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from model import letter_grade


CARD_COLORS = {
    "Excellent": "#22c55e",
    "Good": "#3b82f6",
    "Average": "#f59e0b",
    "Poor": "#ef4444",
}


def build_gauge(value, category):
    """Semi-circular gauge showing predicted grade %, colored by category."""
    color = CARD_COLORS.get(category, "#3b82f6")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "visible": False},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 60], "color": "#fef3c7"},
                {"range": [60, 80], "color": "#dbeafe"},
                {"range": [80, 100], "color": "#dcfce7"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


def scatter_with_trend(df, x_col, y_col, title, x_label, y_label, color="#3b82f6"):
    """Scatter plot with a linear trend line overlay."""
    x = df[x_col].values
    y = df[y_col].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=color, size=7, opacity=0.55),
        name="Students",
    ))

    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = coeffs[0] * x_line + coeffs[1]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=color, width=3),
            name="Trend",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def grade_distribution_bar(df):
    """Bar chart of how many students fall into each letter-grade band."""
    bands = df["final_grade"].apply(letter_grade)
    order = ["A (80-100)", "B (60-79)", "C (40-59)", "D (<40)"]
    counts = bands.value_counts().reindex(order, fill_value=0)
    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        title="Grade Distribution",
        xaxis_title="",
        yaxis_title="Number of Students",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_metric_comparison(metrics_a, metrics_b, label_a="Model A", label_b="Model B"):
    """Grouped bar chart comparing two regression models' metrics."""
    keys = ["r2", "mae", "rmse"]
    display_names = ["R\u00b2 Score", "MAE", "RMSE"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=label_a, x=display_names, y=[metrics_a[k] for k in keys], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name=label_b, x=display_names, y=[metrics_b[k] for k in keys], marker_color="#f59e0b"))
    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_feature_importance_bar(feature_importances: dict):
    names = [k.replace("_", " ").title() for k in feature_importances.keys()]
    values = list(feature_importances.values())
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color="#3b82f6",
    ))
    fig.update_layout(
        title="Feature Importance",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
