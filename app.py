import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip

# --- THEMES & HUMAN TOUCH ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if 'step' not in st.session_state:
    st.session_state.step = "welcome"
if 'config' not in st.session_state:
    st.session_state.config = {}

# --- STEP 1: WELCOME & UPLOAD ---
if st.session_state.step == "welcome":
    st.title("👋 Welcome to Traffic Analyst")
    st.markdown("I'll help you scan your footage and generate a professional traffic report. **Let's start by looking at your video.**")
    
    uploaded_file = st.file_uploader("Drop your video file here", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2: HUMAN QUESTIONS ---
elif st.session_state.step == "briefing":
    st.header("📋 Analysis Briefing")
    st.write("Before I process the footage, how should the report be labeled?")
    
    with st.container(border=True):
        location = st.text_input("Station / Location Name", placeholder="e.g. North Highway - Exit 12")
        duration = st.slider("How many seconds should I analyze?", 1, 60, 10)
        
        st.write("**Visual Settings:**")
        col1, col2 = st.columns(2)
        with col1:
            burn_sidebar = st.toggle("Include HUD sidebar in video", value=True)
            show_overlays = st.toggle("Show AI tracking boxes", value=True)
        with col2:
            show_graph = st.toggle("Include distribution graph", value=True)

    if st.button("🚀 Begin Analysis", use_container_width=True):
        st.session_state.config = {
            "location": location if location else "Traffic Station Alpha",
            "duration": duration,
            "graph": show_graph,
            "burn_sidebar": burn_sidebar,
            "overlays": show_overlays
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3: THE ENGINE ---
elif st.session_state.step == "processing":
    st.header("🧠 Processing Report...")
    st.info(f"Analyzing {st.session_state.config['duration']} seconds for **{st.session_state.config['location']}**.")
    
    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.config['duration'] * fps)

        raw_out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_out_path, fourcc, fps, (w, h))

        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}

        with st.status("Generating High-Definition HUD...", expanded=True) as status:
            results = model.track(source=st.session_state.video_path, stream=True, imgsz=320, persist=True)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.plot() if st.session_state.config['overlays'] else r.orig_img
                current_v = 0 # Reset frame count
                
                # Tally counts
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts:
                        counts[label] += 1
                        current_v += 1

                # --- BURN SIDEBAR INTO VIDEO ---
                if st.session_state.config['burn_sidebar']:
                    # Draw sidebar background
                    sidebar_w = int(w * 0.25)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (15, 15, 15), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Labels & Data
                    cv2.putText(frame, st.session_state.config['location'].upper(), (20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                    
                    cv2.putText(frame, "LIVE TELEMETRY", (w - sidebar_w + 20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
                    
                    y = 120
                    for obj, count in counts.items():
                        # We show total session count or frame count? Usually, users prefer session total
                        cv2.putText(frame, f"{obj.upper()}: {count}", (w - sidebar_w + 20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y += 40
                    
                    status_msg = "CLEAR" if current_v == 0 else "HEAVY" if current_v > 8 else "FLOWING"
                    cv2.putText(frame, f"STATUS: {status_msg}", (w - sidebar_w + 20, h - 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

                out.write(cv2.resize(frame, (w, h)))
            
            out.release()
            cap.release()
            
            # Web Conversion
            final_out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            clip = VideoFileClip(raw_out_path)
            clip.write_videofile(final_out_path, codec="libx264", audio=False)
            status.update(label="Report Finalized!", state="complete")

        # --- RESULTS DISPLAY ---
        st.video(final_out_path)
        
        c1, c2, c3 = st.columns(3)
        with open(final_out_path, 'rb') as f:
            c1.download_button("📥 Download Report", f.read(), f"{st.session_state.config['location']}.mp4", "video/mp4")
        if c2.button("🔄 New Analysis"): reset_app()
        if c3.button("🛑 End Session", type="primary"): reset_app()

    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Return to Start"): reset_app()
