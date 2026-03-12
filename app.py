import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO

# ... (Previous Steps 1-5 stay the same) ...

# --- STEP 6: PROCESSING ---
elif st.session_state.step == "process":
    st.header("⚙️ Generating Traffic Video")
    
    try:
        model = YOLO("yolo11n.pt")
        video_path = st.session_state.video_path
        
        # Setup Video Reader
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.data['duration'] * fps)

        # Setup Video Writer (The 'avc1' codec is key for browsers)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
        
        progress_bar = st.progress(0)
        st_frame = st.empty() # Placeholder for live preview

        # Process frame by frame
        results = model.track(source=video_path, stream=True, imgsz=320, persist=True)
        
        for i, r in enumerate(results):
            if i >= max_frames:
                break
            
            # 1. Get annotated frame
            if st.session_state.data['overlays']:
                annotated_frame = r.plot() # YOLO draws the boxes for us
            else:
                annotated_frame = r.orig_img # Original frame with no boxes
            
            # 2. Write frame to file
            out.write(annotated_frame)
            
            # 3. Update Counts
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in counts: counts[label] += 1
            
            # 4. Update UI
            progress_bar.progress((i + 1) / max_frames)

        out.release()
        cap.release()

        # --- DISPLAY FINAL VIDEO ---
        st.success("Analysis Complete!")
        
        # Show the video player
        with open(output_path, 'rb') as v_file:
            video_bytes = v_file.read()
        st.video(video_bytes)

        # --- Sidebar / Results Logic ---
        v_total = counts['car'] + counts['bus'] + counts['truck']
        status = "Flowing" if v_total < (10 * st.session_state.data['duration']) else "Heavy"
        
        if st.session_state.data['sidebar']:
            st.sidebar.title("Live Report")
            st.sidebar.metric("Status", status)
            for k, v in counts.items():
                st.sidebar.write(f"{k.capitalize()}s: {v}")

    except Exception as e:
        st.error(f"Video Generation Error: {e}")

    if st.button("Start New Analysis"):
        st.session_state.step = "upload"
        st.rerun()
