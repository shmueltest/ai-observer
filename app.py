import streamlit as st
import tempfile
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# --- Initialization ---
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'settings' not in st.session_state:
    st.session_state.settings = {}

st.set_page_config(page_title="Traffic Intel Pro", layout="wide")
st.title("🚦 Traffic Analytics System")

# --- STEP 1: UPLOAD ---
if st.session_state.step == "upload":
    uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        st.session_state.video_data = uploaded_file.getvalue()
        st.session_state.step = "duration"
        st.rerun()

# --- STEP 2: DURATION ---
if st.session_state.step == "duration":
    st.subheader("Analysis Parameters")
    duration = st.number_input("How many seconds to analyze?", min_value=1, max_value=300, value=10)
    if st.button("Set Duration"):
        st.session_state.settings['duration'] = duration
        st.session_state.step = "graph_ask"
        st.rerun()

# --- STEP 3: GRAPH OPTION ---
if st.session_state.step == "graph_ask":
    st.subheader("Visualization Settings")
    st.write("Generate a results graph (volume over time)?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="graph_yes"):
            st.session_state.settings['graph'] = True
            st.session_state.step = "overlay_ask"
            st.rerun()
    with col2:
        if st.button("No", key="graph_no"):
            st.session_state.settings['graph'] = False
            st.session_state.step = "overlay_ask"
            st.rerun()

# --- STEP 4: OVERLAY OPTION ---
if st.session_state.step == "overlay_ask":
    st.subheader("Output Video Style")
    st.write("Include AI bounding box overlays on the output video?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("With Overlays"):
            st.session_state.settings['overlays'] = True
            st.session_state.step = "sidebar_ask"
            st.rerun()
    with col2:
        if st.button("Without Overlays"):
            st.session_state.settings['overlays'] = False
            st.session_state.step = "sidebar_ask"
            st.rerun()

# --- STEP 5: SIDEBAR OPTION ---
if st.session_state.step == "sidebar_ask":
    st.subheader("Data Display")
    st.write("Include a data sidebar with live counts and traffic status?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Include Sidebar"):
            st.session_state.settings['sidebar'] = True
            st.session_state.step = "process"
            st.rerun()
    with col2:
        if st.button("No Sidebar"):
            st.session_state.settings['sidebar'] = False
            st.session_state.step = "process"
            st.rerun()

# --- STEP 6: PROCESSING & RESULTS ---
if st.session_state.step == "process":
    st.header("⚙️ Processing Traffic Data")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(st.session_state.video_data)
        video_path = tmp.name

    # Load YOLO
    model = YOLO("yolo11n.pt") 
    
    # Process Video Logic
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_process = int(st.session_state.settings['duration'] * fps)
    
    counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
    
    with st.spinner("Analyzing frames..."):
        # We simulate frame analysis for the selected duration
        results = model.predict(source=video_path, frames=frames_to_process, imgsz=320, verbose=False)
        
        # Aggregate counts from the last processed frame for the status
        last_boxes = results[-1].boxes
        for box in last_boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in counts:
                counts[name] += 1

    # --- RESULTS CELL ---
    st.divider()
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.subheader("Analysis Result")
        # Logic for Traffic Status
        total_vehicles = counts['car'] + counts['bus'] + counts['truck']
        if total_vehicles == 0: status = "Clear"
        elif total_vehicles < 5: status = "Flowing"
        elif total_vehicles < 12: status = "Heavy"
        else: status = "Stopped/Congested"
        
        st.info(f"Traffic Status: **{status}**")
        
        if st.session_state.settings['graph']:
            st.line_chart(pd.DataFrame([counts]).T)

    if st.session_state.settings['sidebar']:
        with res_col2:
            st.sidebar.header("📊 Live Count")
            for item, count in counts.items():
                st.sidebar.write(f"{item.capitalize()}s: {count}")
            st.sidebar.write(f"**Status:** {status}")

    if st.button("Start New Analysis"):
        st.session_state.step = "upload"
        st.rerun()
