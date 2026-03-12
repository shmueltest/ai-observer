import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO

# --- APP SETUP ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'data' not in st.session_state:
    st.session_state.data = {}

# --- STEP 1: UPLOAD ---
if st.session_state.step == "upload":
    st.title("🚦 Traffic Analytics System")
    uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "duration"
        st.rerun()

# --- STEP 2: DURATION ---
elif st.session_state.step == "duration":
    st.header("Analysis Parameters")
    duration = st.number_input("How many seconds to analyze?", min_value=1, max_value=60, value=5)
    if st.button("Confirm Duration"):
        st.session_state.data['duration'] = duration
        st.session_state.step = "graph_ask"
        st.rerun()

# --- STEP 3: GRAPH ASK ---
elif st.session_state.step == "graph_ask":
    st.header("Visualizations")
    st.write("Generate a results graph?")
    col1, col2 = st.columns(2)
    if col1.button("Yes"):
        st.session_state.data['graph'] = True
        st.session_state.step = "overlay_ask"; st.rerun()
    if col2.button("No"):
        st.session_state.data['graph'] = False
        st.session_state.step = "overlay_ask"; st.rerun()

# --- STEP 4: OVERLAY ASK ---
elif st.session_state.step == "overlay_ask":
    st.header("Output Style")
    st.write("Include AI bounding box overlays?")
    col1, col2 = st.columns(2)
    if col1.button("With Overlays"):
        st.session_state.data['overlays'] = True
        st.session_state.step = "sidebar_ask"; st.rerun()
    if col2.button("No Overlays"):
        st.session_state.data['overlays'] = False
        st.session_state.step = "sidebar_ask"; st.rerun()

# --- STEP 5: SIDEBAR ASK ---
elif st.session_state.step == "sidebar_ask":
    st.header("Interface Layout")
    st.write("Display real-time count in a side bar?")
    col1, col2 = st.columns(2)
    if col1.button("Include Sidebar"):
        st.session_state.data['sidebar'] = True
        st.session_state.step = "process"; st.rerun()
    if col2.button("Main View Only"):
        st.session_state.data['sidebar'] = False
        st.session_state.step = "process"; st.rerun()

# --- STEP 6: PROCESSING ---
elif st.session_state.step == "process":
    st.header("⚙️ Processing Traffic Data")
    
    try:
        model = YOLO("yolo11n.pt")
        video_path = st.session_state.video_path
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.data['duration'] * fps)
        
        # Tracking counts
        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
        north_bound = 0
        south_bound = 0

        with st.spinner(f"Analyzing {st.session_state.data['duration']} seconds of footage..."):
            # Fixed YOLO prediction loop using stream=True
            results = model.track(source=video_path, stream=True, imgsz=320, verbose=False, persist=True)
            
            for i, r in enumerate(results):
                if i >= max_frames:
                    break
                
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts:
                        counts[label] += 1
                        
                        # North/South Logic (Simplified by Y-coordinate of detection)
                        # Top half of frame = North, Bottom half = South
                        y_coord = box.xyxy[0][1]
                        if y_coord < (height / 2):
                            north_bound += 1
                        else:
                            south_bound += 1
        
        # Traffic Status Logic
        v_total = counts['car'] + counts['bus'] + counts['truck']
        if v_total == 0: status = "Clear"
        elif v_total < (5 * st.session_state.data['duration']): status = "Flowing"
        elif v_total < (15 * st.session_state.data['duration']): status = "Heavy"
        else: status = "Stopped/Congested"

        # --- Sidebar Display ---
        if st.session_state.data['sidebar']:
            st.sidebar.title("Live Traffic Report")
            st.sidebar.info(f"Status: {status}")
            st.sidebar.write(f"⬆️ North Bound: {north_bound}")
            st.sidebar.write(f"⬇️ South Bound: {south_bound}")
            st.sidebar.divider()
            for k, v in counts.items():
                st.sidebar.write(f"{k.capitalize()}s: {v}")

        # --- Main Results ---
        st.success("Analysis Complete")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Traffic Metrics")
            st.write(f"**Overall Assessment:** {status}")
            st.write(f"**North Bound Activity:** {north_bound} detections")
            st.write(f"**South Bound Activity:** {south_bound} detections")

        if st.session_state.data['graph']:
            with res_col2:
                st.subheader("Vehicle Breakdown")
                df = pd.DataFrame(list(counts.items()), columns=['Vehicle', 'Count'])
                st.bar_chart(df.set_index('Vehicle'))

    except Exception as e:
        st.error(f"Application Error: {e}")

    if st.button("Start New Analysis"):
        st.session_state.step = "upload"
        st.rerun()
