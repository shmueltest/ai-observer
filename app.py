import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO

# --- APP SETUP ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

# Initialize Session State Variables
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'data' not in st.session_state:
    st.session_state.data = {}

# --- STEP 1: UPLOAD ---
if st.session_state.step == "upload":
    st.title("🚦 Traffic Analytics System")
    uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        # Save file to a temporary location that persists across reruns
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "duration"
        st.rerun()

# --- STEP 2: DURATION ---
elif st.session_state.step == "duration":
    st.header("Step 1: Analysis Timing")
    duration = st.number_input("How many seconds to analyze?", min_value=1, max_value=60, value=5)
    if st.button("Confirm Duration"):
        st.session_state.data['duration'] = duration
        st.session_state.step = "graph_ask"
        st.rerun()

# --- STEP 3: GRAPH ASK ---
elif st.session_state.step == "graph_ask":
    st.header("Step 2: Data Visualization")
    st.write("Would you like to generate a results graph?")
    col1, col2 = st.columns(2)
    if col1.button("Yes"):
        st.session_state.data['graph'] = True
        st.session_state.step = "overlay_ask"; st.rerun()
    if col2.button("No"):
        st.session_state.data['graph'] = False
        st.session_state.step = "overlay_ask"; st.rerun()

# --- STEP 4: OVERLAY ASK ---
elif st.session_state.step == "overlay_ask":
    st.header("Step 3: Visual Style")
    st.write("Generate output video with AI bounding box overlays?")
    col1, col2 = st.columns(2)
    if col1.button("With Overlays"):
        st.session_state.data['overlays'] = True
        st.session_state.step = "sidebar_ask"; st.rerun()
    if col2.button("Without Overlays"):
        st.session_state.data['overlays'] = False
        st.session_state.step = "sidebar_ask"; st.rerun()

# --- STEP 5: SIDEBAR ASK ---
elif st.session_state.step == "sidebar_ask":
    st.header("Step 4: Layout")
    st.write("Display real-time count in a side bar?")
    col1, col2 = st.columns(2)
    if col1.button("Include Sidebar"):
        st.session_state.data['sidebar'] = True
        st.session_state.step = "process"; st.rerun()
    if col2.button("Main View Only"):
        st.session_state.data['sidebar'] = False
        st.session_state.step = "process"; st.rerun()

# --- STEP 6: PROCESSING & RESULTS ---
elif st.session_state.step == "process":
    st.header("⚙️ Processing Traffic Data")
    
    try:
        # Load Nano model for speed
        model = YOLO("yolo11n.pt")
        
        # Open video and get stats
        video_path = st.session_state.video_path
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames_to_process = int(st.session_state.data['duration'] * fps)
        
        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
        
        with st.spinner("AI scanning video..."):
            # We only process the requested duration
            results = model.predict(source=video_path, frames=frames_to_process, imgsz=320, verbose=False)
            
            # Count the final state of objects
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts:
                        counts[label] += 1
        
        # Traffic Status Logic
        total_v = counts['car'] + counts['bus'] + counts['truck']
        if total_v == 0: status = "Clear"
        elif total_v < 8: status = "Flowing"
        elif total_v < 20: status = "Heavy"
        else: status = "Stopped/Heavy Congestion"

        # --- Display Side Bar ---
        if st.session_state.data['sidebar']:
            st.sidebar.title("Traffic Report")
            st.sidebar.subheader(f"Status: {status}")
            for item, count in counts.items():
                st.sidebar.write(f"{item.capitalize()}s: {count}")

        # --- Main Results Cell ---
        st.success("Analysis Complete")
        st.markdown(f"### Final Traffic Assessment: **{status}**")
        
        if st.session_state.data['graph']:
            st.subheader("Vehicle Distribution")
            df = pd.DataFrame(list(counts.items()), columns=['Vehicle Type', 'Count'])
            st.bar_chart(df.set_index('Vehicle Type'))

    except Exception as e:
        st.error(f"Application Error: {e}")

    if st.button("Restart New Session"):
        st.session_state.step = "upload"
        st.rerun()
