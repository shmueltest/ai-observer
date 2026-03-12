import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip

# --- SESSION RESET ---
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'data' not in st.session_state:
    st.session_state.data = {}

# --- UI LOGIC ---
st.title("🚦 Traffic Analytics System")

if st.session_state.step == "upload":
    uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "duration"
        st.rerun()

elif st.session_state.step == "duration":
    st.header("Step 1: Duration")
    duration = st.number_input("Seconds to analyze?", 1, 60, 5)
    if st.button("Confirm"):
        st.session_state.data['duration'] = duration
        st.session_state.step = "graph_ask"; st.rerun()

elif st.session_state.step == "graph_ask":
    st.header("Step 2: Graphs")
    if st.button("Yes"): 
        st.session_state.data['graph'] = True
        st.session_state.step = "overlay_ask"; st.rerun()
    if st.button("No"): 
        st.session_state.data['graph'] = False
        st.session_state.step = "overlay_ask"; st.rerun()

elif st.session_state.step == "overlay_ask":
    st.header("Step 3: Overlays")
    if st.button("With AI Boxes"):
        st.session_state.data['overlays'] = True
        st.session_state.step = "sidebar_ask"; st.rerun()
    if st.button("Original Video Only"):
        st.session_state.data['overlays'] = False
        st.session_state.step = "sidebar_ask"; st.rerun()

elif st.session_state.step == "sidebar_ask":
    st.header("Step 4: Layout")
    if st.button("Include Sidebar"):
        st.session_state.data['sidebar'] = True
        st.session_state.step = "process"; st.rerun()
    if st.button("Main View Only"):
        st.session_state.data['sidebar'] = False
        st.session_state.step = "process"; st.rerun()

elif st.session_state.step == "process":
    st.header("⚙️ Processing...")
    
    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.data['duration'] * fps)

        # Raw OpenCV Output (Often unplayable by browsers)
        raw_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Reliable for raw writing
        out = cv2.VideoWriter(raw_output, fourcc, fps, (w, h))

        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}

        with st.status("Analyzing Frames...") as status:
            results = model.track(source=st.session_state.video_path, stream=True, imgsz=320)
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                # Get annotated frame and RESIZE to match Writer dimensions
                res_frame = r.plot() if st.session_state.data['overlays'] else r.orig_img
                res_frame = cv2.resize(res_frame, (w, h))
                out.write(res_frame)

                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in counts: counts[label] += 1
            
            out.release()
            cap.release()
            status.update(label="Converting for Web Playback...", state="running")

            # --- THE MAGIC FIX: Convert to H.264 ---
            final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            clip = VideoFileClip(raw_output)
            clip.write_videofile(final_output, codec="libx264", audio=False)
            clip.close()
            status.update(label="Done!", state="complete")

        # Results Display
        with open(final_output, 'rb') as f:
            v_bytes = f.read()

        st.video(v_bytes)
        st.download_button("📥 Download Video", v_bytes, "traffic_results.mp4", "video/mp4")
        
        if st.button("🔄 Analyze New Link/Video"): reset_session()
        if st.button("🛑 End Session", type="primary"): reset_session()

        if st.session_state.data['sidebar']:
            for k, v in counts.items(): st.sidebar.write(f"{k}: {v}")

    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Reset App"): reset_session()
