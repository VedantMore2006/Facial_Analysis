# ============================================================================
# STREAMLIT DASHBOARD FOR BEHAVIORAL ANALYSIS VISUALIZATION
# ============================================================================
# Purpose: Interactive web dashboard to visualize facial analysis features
# Input: CSV file containing scaled feature values (0-1 range)
# Output: Time-series plots, heatmaps, and summary statistics
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page layout to use full width for better visualization
st.set_page_config(layout="wide")

# Main dashboard title
st.title("Behavioral Analysis Dashboard")

# File uploader widget - accepts CSV files containing feature data
uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

# ============================================================================
# MAIN VISUALIZATION LOGIC (only runs when file is uploaded)
# ============================================================================
if uploaded_file is not None:

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display first few rows for quick data inspection
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Extract all column names as feature names
    feature_names = df.columns.tolist()

    # ----------------------------
    # Baseline Configuration Section
    # ----------------------------
    # Allow user to specify baseline duration and FPS for visualization context
    baseline_seconds = st.number_input("Baseline Duration (seconds)", value=30)
    fps = st.number_input("FPS", value=15)

    # Calculate how many frames belong to baseline phase (shown as gray region)
    baseline_frames = int(baseline_seconds * fps)

    # ----------------------------
    # Time-Series Line Plot Section
    # ----------------------------
    # Shows feature values over time, allowing identification of behavioral patterns
    st.subheader("Scaled Feature Signals")

    # Multi-select dropdown to choose which features to visualize
    selected_features = st.multiselect(
        "Select features to display",
        feature_names,
        default=feature_names
    )

    # Create interactive Plotly figure for time-series visualization
    fig = go.Figure()

    # Add a line trace for each selected feature
    for feature in selected_features:
        fig.add_trace(go.Scatter(
            y=df[feature],
            mode='lines',
            name=feature
        ))

    # Add gray shaded region to indicate baseline collection phase
    fig.add_vrect(
        x0=0,
        x1=baseline_frames,
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

    # Configure plot layout with appropriate labels
    fig.update_layout(
        height=400,
        xaxis_title="Frame",
        yaxis_title="Scaled Value (0–1)"
    )

    # Display the plot in Streamlit (using full container width)
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Behavioral Heatmap Section
    # ----------------------------
    # Visualize intensity patterns across features and time using color intensity
    st.subheader("Behavioral Heatmap")

    # Transpose data so features are rows and frames are columns
    heatmap_data = df[selected_features].T

    # Create heatmap with red-blue color scale (0=blue, 1=red)
    heatmap_fig = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=0,  # Minimum value (blue)
        zmax=1   # Maximum value (red)
    )

    # Display heatmap in full width
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # ----------------------------
    # Summary Statistics Section
    # ----------------------------
    # Provide quick statistical overview of all features in the session
    st.subheader("Session Summary Statistics")

    # Calculate mean and standard deviation for each feature
    summary = pd.DataFrame({
        "Mean": df.mean(),
        "Std": df.std()
    })

    # Display statistics table
    st.dataframe(summary)

# Display info message when no file is uploaded
else:
    st.info("Upload a feature CSV file to visualize.")