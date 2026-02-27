import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Behavioral Analysis Dashboard")

uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    feature_names = df.columns.tolist()

    # ----------------------------
    # Baseline Input
    # ----------------------------
    baseline_seconds = st.number_input("Baseline Duration (seconds)", value=30)
    fps = st.number_input("FPS", value=15)

    baseline_frames = int(baseline_seconds * fps)

    # ----------------------------
    # Time-Series Plot
    # ----------------------------
    st.subheader("Scaled Feature Signals")

    selected_features = st.multiselect(
        "Select features to display",
        feature_names,
        default=feature_names
    )

    fig = go.Figure()

    for feature in selected_features:
        fig.add_trace(go.Scatter(
            y=df[feature],
            mode='lines',
            name=feature
        ))

    fig.add_vrect(
        x0=0,
        x1=baseline_frames,
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

    fig.update_layout(
        height=400,
        xaxis_title="Frame",
        yaxis_title="Scaled Value (0–1)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Heatmap
    # ----------------------------
    st.subheader("Behavioral Heatmap")

    heatmap_data = df[selected_features].T

    heatmap_fig = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=0,
        zmax=1
    )

    st.plotly_chart(heatmap_fig, use_container_width=True)

    # ----------------------------
    # Summary Statistics
    # ----------------------------
    st.subheader("Session Summary Statistics")

    summary = pd.DataFrame({
        "Mean": df.mean(),
        "Std": df.std()
    })

    st.dataframe(summary)

else:
    st.info("Upload a feature CSV file to visualize.")