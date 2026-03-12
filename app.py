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
    st.markdown("I'll help you scan your footage and generate a detailed traffic report. **Let's start by looking at your video.**")
    
    uploaded_file = st.file_uploader("Drop your video file here", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2: HUMAN QUESTIONS (The Briefing) ---
elif st.session_state.step == "briefing":
    st.header("📋 Analysis Briefing")
    st.write("Before I start the engine, tell me how you want the final report to look.")
    
    with st.expander("Configure Analysis Details", expanded=True):
        duration = st.slider("How many seconds should I analyze?", 1, 60, 10)
        st.write("---")
        
        st.write("**Visual Preferences:**")
        show_graph = st.checkbox("Generate a vehicle distribution graph?", value=True)
        burn_sidebar = st.checkbox("Burn the 'Live Count' sidebar directly into the video file?", value=True)
        show_overlays = st.checkbox("Include AI bounding boxes (Overlays)?", value=True)

    if st.button("🚀 Start Analysis"):
        st.session_state.config = {
            "duration": duration,
            "graph": show_graph,
            "burn_sidebar": burn_sidebar,
            "overlays": show_overlays
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3: THE WORKHORSE ---
elif st.session_state.step == "processing":
    st.header("🧠 Thinking...")
    st.info("I'm scanning the road and drawing your report. This will just take a moment.")
    
    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.config['duration'] * fps)

        # Temp files for processing
        raw_out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_out_path, fourcc, fps, (w, h))

        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}

        with st.status("Analyzing and drawing HUD...") as status:
            results = model.track(source=st.session_state.video_path, stream=True, imgsz=320)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                # 1. Base frame (Plot overlays if requested)
                frame = r.plot() if st.session_state.config['overlays'] else r.orig_img
                
                # 2. Track counts
                current_frame_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts:
                        counts[label] += 1
                        current_frame_counts[label] += 1

                # 3. Burn Sidebar into Video (HUD)
                if st.session_state.config['burn_sidebar']:
                    # Draw semi-transparent black overlay for sidebar area
                    sidebar_w = int(w * 0.25)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # Draw Text
                    cv2.putText(frame, "TRAFFIC REPORT", (w - sidebar_w + 20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    y_offset = 120
                    for obj, count in current_frame_counts.items():
                        cv2.putText(frame, f"{obj.upper()}S: {count}", (w - sidebar_w + 20, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        y_offset += 40
                    
                    # Traffic Status Logic
                    v_total = sum(current_frame_counts.values())
                    status_text = "CLEAR" if v_total == 0 else "HEAVY" if v_total > 10 else "FLOWING"
                    cv2.putText(frame, f"STATUS: {status_text}", (w - sidebar_w + 20, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                out.write(cv2.resize(frame, (w, h)))
            
            out.release()
            cap.release()
            
            # Final conversion for Web Playback
            final_out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            clip = VideoFileClip(raw_out_path)
            clip.write_videofile(final_out_path, codec="libx264", audio=False)
            status.update(label="Report Generated!", state="complete")

        # --- FINAL DISPLAY ---
        with open(final_out_path, 'rb') as f:
            video_bytes = f.read()
        
        st.video(video_bytes)
        
        c1, c2, c3 = st.columns(3)
        c1.download_button("📥 Download Official Report", video_bytes, "traffic_report.mp4", "video/mp4")
        if c2.button("🔄 Analyze New Video"): reset_app()
        if c3.button("🛑 Clear Session", type="primary"): reset_app()

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        if st.button("Take me back"): reset_app()
