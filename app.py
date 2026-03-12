import streamlit as st
import tempfile
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Initialize Session States
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'settings' not in st.session_state:
    st.session_state.settings = {}

st.set_page_config(page_title="Traffic Analytics", layout="wide")

# --- STEP 1: UPLOAD ---
if st.session_state.step == "upload":
    st.title("🚦 Traffic Analytics System")
    uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        # We store the file in a temp directory immediately to avoid memory bloat
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.temp_video_path = tfile.name
        st.session_state.step = "duration"
        st.rerun()

# --- STEP 2: DURATION ---
elif st.session_state.step == "duration":
    st.header("Analysis Duration")
    seconds = st.number_input("How many seconds to analyze?", min_value=1, max_value=60, value=5)
    if st.button("Confirm Duration"):
        st.session_state.settings['duration'] = seconds
        st.session_state.step = "graph_ask"
        st.rerun()

# --- STEP 3: GRAPH ASK ---
elif st.session_state.step == "graph_ask":
    st.header("Visualizations")
    st.write("Generate a results graph?")
    c1, c2 = st.columns(2)
    if c1.button("Yes"):
        st.session_state.settings['graph'] = True
        st.session_state.step = "overlay_ask"; st.rerun()
    if c2.button("No"):
        st.session_state.settings['graph'] = False
        st.session_state.step = "overlay_ask"; st.rerun()

# --- STEP 4: OVERLAY ASK ---
elif st.session_state.step == "overlay_ask":
    st.header("Video Style")
    st.write("Include AI overlays (boxes/labels)?")
    c1, c2 = st.columns(2)
    if c1.button("With Overlays"):
        st.session_state.settings['overlays'] = True
        st.session_state.step = "sidebar_ask"; st.rerun()
    if c2.button("No Overlays"):
        st.session_state.settings['overlays'] = False
        st.session_state.step = "sidebar_ask"; st.rerun()

# --- STEP 5: SIDEBAR ASK ---
elif st.session_state.step == "sidebar_ask":
    st.header("Data Display")
    st.write("Display count data in a sidebar?")
    c1, c2 = st.columns(2)
    if c1.button("Include Sidebar"):
        st.session_state.settings['sidebar'] = True
        st.session_state.step = "process"; st.rerun()
    if c2.button("Main View Only"):
        st.session_state.settings['sidebar'] = False
        st.session_state.step = "process"; st.rerun()

# --- STEP 6: PROCESSING ---
elif st.session_state.step == "process":
    st.header("⚙️ Processing...")
    
    try:
        model = YOLO("yolo11n.pt")
        video_path = st.session_state.temp_video_path
        
        # Open video to get stats
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(st.session_state.settings['duration'] * fps)
        
        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
        
        # Run YOLO Analysis
        results = model.predict(source=video_path, frames=total_frames, imgsz=320, verbose=False)
        
        # Aggregate the data
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in counts:
                    counts[label] += 1
        
        # Logic for Traffic Status
        v_total = counts['car'] + counts['bus'] + counts['truck']
        if v_total == 0: status = "Clear"
        elif v_total < 10: status = "Flowing"
        elif v_total < 30: status = "Heavy"
        else: status = "Stopped"

        # --- Display Results ---
        if st.session_state.settings['sidebar']:
            st.sidebar.title("Live Metrics")
            st.sidebar.metric("Traffic Status", status)
            for k, v in counts.items():
                st.sidebar.write(f"**{k.capitalize()}s:** {v}")
        
        st.success("Analysis Complete")
        st.write(f"### Final Status: {status}")
        
        if st.session_state.settings['graph']:
            df = pd.DataFrame(list(counts.items()), columns=['Object', 'Count'])
            st.bar_chart(df.set_index('Object'))

    except Exception as e:
        st.error(f"Error during processing: {e}")
    
    if st.button("New Analysis"):
        st.session_state.step = "upload"
        st.rerun()
