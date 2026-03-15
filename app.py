import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np

# --- DARK MODE UI & STYLING ---
st.set_page_config(page_title="Traffic Tracker", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #464b5d; }
    .stButton>button:hover { border: 1px solid #ff4b4b; color: #ff4b4b; }
    [data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if 'step' not in st.session_state:
    st.session_state.step = "welcome"
if 'config' not in st.session_state:
    st.session_state.config = {}

# --- STEP 1: WELCOME ---
if st.session_state.step == "welcome":
    st.title("🚦 Traffic Tracking System")
    st.subheader("שמואל קוימאן וישי גפני")
    
    uploaded_file = st.file_uploader("Upload footage for analysis", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2: BRIEFING ---
elif st.session_state.step == "briefing":
    st.header("?מה מרצה לעשות - What do You Want to Do?")
    
    with st.container(border=True):
        loc = st.text_input("Location Label", "Sector 7G - Main Intersection")
        duration = st.slider("Analysis Window (Seconds)", 1, 60, 10)
        
        c1, c2 = st.columns(2)
        with c1:
            burn_hud = st.toggle("Create new video with information? - ?להוסיף את המידע לסרטון", value=True)
            overlays = st.toggle("Object Tracking Boxes - תיבות מעקב", value=True)
        with c2:
            gen_graph = st.toggle("Generate Final Metrics Graph - ייצור גרף של מה היה בסרטון", value=True)
            use_sidebar = st.toggle("Live Tracking metrics in Sidebar - מדד זמן אמת של הכביש", value=True)

    if st.button("🚀 INITIATE ANALYSIS בוא נתחיל"):
        st.session_state.config = {
            "loc": loc, "duration": duration, "burn": burn_hud, 
            "overlays": overlays, "graph": gen_graph, "sidebar": use_sidebar
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3: PROCESSING ---
elif st.session_state.step == "processing":
    st.header("🧠 Processing Intelligence...")
    
    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.config['duration'] * fps)

        raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        tracked_ids = set() 
        entry_points = {} # To track direction
        final_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        direction_counts = {"Inbound": 0, "Outbound": 0}
        
        with st.status("Analyzing unique vehicle signatures and directions...", expanded=True) as status:
            results = model.track(source=st.session_state.video_path, stream=True, persist=True, imgsz=320)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.orig_img.copy()
                
                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confidences):
                        label = model.names[cls]
                        if label in final_counts:
                            # Percentage formatting for the box label
                            conf_per = int(conf * 100)
                            
                            # Logic for Direction
                            y_center = (box[1] + box[3]) / 2
                            if obj_id not in entry_points:
                                entry_points[obj_id] = y_center
                                final_counts[label] += 1
                                tracked_ids.add(obj_id)
                            else:
                                if y_center > entry_points[obj_id] + 10: # Moved down
                                    direction_counts["Inbound"] += 1
                                    entry_points[obj_id] = 99999 # Marked as counted
                                elif y_center < entry_points[obj_id] - 10: # Moved up
                                    direction_counts["Outbound"] += 1
                                    entry_points[obj_id] = -99999 # Marked as counted

                            # Manual Plotting with % Confidence
                            if st.session_state.config['overlays']:
                                x1, y1, x2, y2 = map(int, box)
                                color = (0, 255, 0) # Green for all
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, f"{label} {conf_per}%", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- HUD RENDERING ---
                if st.session_state.config['burn']:
                    sidebar_w = int(w * 0.25)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, st.session_state.config['loc'].upper(), (20, 50), 2, 0.8, (255,255,255), 2)
                    y = 120
                    for obj, val in final_counts.items():
                        cv2.putText(frame, f"{obj.upper()}: {val}", (w - sidebar_w + 20, y), 2, 0.6, (0, 255, 0), 1)
                        y += 40
                    
                    cv2.putText(frame, "DIRECTIONS:", (w - sidebar_w + 20, y+20), 2, 0.6, (255, 255, 0), 1)
                    cv2.putText(frame, f"IN: {direction_counts['Inbound']}", (w - sidebar_w + 20, y+60), 2, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"OUT: {direction_counts['Outbound']}", (w - sidebar_w + 20, y+100), 2, 0.6, (255, 255, 255), 1)

                out.write(cv2.resize(frame, (w, h)))
            
            out.release()
            cap.release()
            
            final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            clip = VideoFileClip(raw_path)
            clip.write_videofile(final_path, codec="libx264", audio=False)
            status.update(label="Analysis Complete", state="complete")

        # --- RESULTS DISPLAY ---
        st.divider()
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.video(final_path)
            with open(final_path, 'rb') as f:
                st.download_button("📥 Download Report", f.read(), "report.mp4", "video/mp4")

        with col_right:
            if st.session_state.config['graph']:
                st.write("### 📊 Distribution")
                df = pd.DataFrame(list(final_counts.items()), columns=['Vehicle', 'Total'])
                st.bar_chart(df.set_index('Vehicle'))
            
            st.write("### ⚡ Actions")
            if st.button("🔄 New Analysis החל מחדש"): reset_app()
            if st.button("🛑 End Session יציאה", type="primary"): reset_app()

        # Sidebar Live Status
        if st.session_state.config['sidebar']:
            st.sidebar.title("Live Directional Stats")
            st.sidebar.metric("Unique Vehicles", len(tracked_ids))
            st.sidebar.divider()
            st.sidebar.metric("Inbound (Down)", direction_counts["Inbound"])
            st.sidebar.metric("Outbound (Up)", direction_counts["Outbound"])

    except Exception as e:
        st.error(f"System Error: {e}")
        if st.button("Emergency Reset"): reset_app()
