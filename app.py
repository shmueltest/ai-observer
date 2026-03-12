import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO

# --- APP CONFIG & SESSION INITIALIZATION ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

# This resets the app and clears all variables
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Default State Initialization
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'data' not in st.session_state:
    st.session_state.data = {}

# --- STEP 1: UPLOAD ---
if st.session_state.step == "upload":
    st.title("🚦 Traffic Analytics System")
    st.write("Upload a traffic feed video to begin the multi-step analysis.")
    uploaded_file = st.file_uploader("Select Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "duration"
        st.rerun()

# --- STEP 2: DURATION ---
elif st.session_state.step == "duration":
    st.header("Step 1: Analysis Timing")
    duration = st.number_input("How many seconds of the video should be analyzed?", 1, 60, 5)
    if st.button("Confirm Duration"):
        st.session_state.data['duration'] = duration
        st.session_state.step = "graph_ask"
        st.rerun()

# --- STEP 3: GRAPH ASK ---
elif st.session_state.step == "graph_ask":
    st.header("Step 2: Visualizations")
    st.write("Generate a vehicle distribution graph?")
    c1, c2 = st.columns(2)
    if c1.button("Yes"):
        st.session_state.data['graph'] = True
        st.session_state.step = "overlay_ask"; st.rerun()
    if c2.button("No"):
        st.session_state.data['graph'] = False
        st.session_state.step = "overlay_ask"; st.rerun()

# --- STEP 4: OVERLAY ASK ---
elif st.session_state.step == "overlay_ask":
    st.header("Step 3: AI Overlays")
    st.write("Draw AI bounding boxes and labels on the result video?")
    c1, c2 = st.columns(2)
    if c1.button("Yes, add Overlays"):
        st.session_state.data['overlays'] = True
        st.session_state.step = "sidebar_ask"; st.rerun()
    if c2.button("No, keep video clean"):
        st.session_state.data['overlays'] = False
        st.session_state.step = "sidebar_ask"; st.rerun()

# --- STEP 5: SIDEBAR ASK ---
elif st.session_state.step == "sidebar_ask":
    st.header("Step 4: Layout")
    st.write("Show counts and traffic status in the sidebar?")
    c1, c2 = st.columns(2)
    if c1.button("Include Sidebar"):
        st.session_state.data['sidebar'] = True
        st.session_state.step = "process"; st.rerun()
    if c2.button("Main Screen Only"):
        st.session_state.data['sidebar'] = False
        st.session_state.step = "process"; st.rerun()

# --- STEP 6: PROCESSING & FINAL RESULTS ---
elif st.session_state.step == "process":
    st.header("⚙️ Processing Traffic Intel...")
    
    try:
        model = YOLO("yolo11n.pt")
        video_path = st.session_state.video_path
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.data['duration'] * fps)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
        
        with st.status("Analyzing and Encoding...", expanded=True) as status_box:
            # Note: stream=True is required to loop through frames manually
            results = model.track(source=video_path, stream=True, imgsz=320, persist=True)
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.plot() if st.session_state.data['overlays'] else r.orig_img
                out.write(frame)
                
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts: counts[label] += 1
            
            out.release()
            cap.release()
            status_box.update(label="Analysis Ready", state="complete")

        # Display Final Video
        st.divider()
        with open(output_path, 'rb') as f:
            v_bytes = f.read()
        
        col_vid, col_btns = st.columns([3, 1])
        with col_vid:
            st.video(v_bytes)
        with col_btns:
            st.download_button("📥 Download MP4", v_bytes, "traffic_analysis.mp4", "video/mp4", use_container_width=True)
            
            # --- THE END SESSION OPTIONS ---
            st.divider()
            if st.button("🔄 Analyze New Video", use_container_width=True):
                reset_session()
            if st.button("🛑 End Session & Clear", use_container_width=True, type="primary"):
                reset_session()

        # Traffic Assessment
        v_total = sum([counts['car'], counts['bus'], counts['truck']])
        status = "Clear" if v_total == 0 else "Heavy" if v_total > 20 else "Flowing"
        
        if st.session_state.data['sidebar']:
            st.sidebar.subheader(f"Status: {status}")
            st.sidebar.write(f"Cars: {counts['car']} | Trucks: {counts['truck']}")

    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Back to Home"): reset_session()
