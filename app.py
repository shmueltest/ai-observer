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

def get_traffic_state(avg_speed):
    """Determines state based on speed thresholds (km/h)"""
    if avg_speed == 0: return "⚪ NO DATA"
    if avg_speed < 15: return "🔴 HEAVY (Congested)"
    if avg_speed < 40: return "🟡 MODERATE (Slow)"
    return "🟢 LIGHT (Free Flow)"

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
            use_sidebar = st.toggle("Live Status & Metrics in Sidebar - מדד זמן אמת של הכביש", value=True)

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
        entry_points = {} # Tracks start Y for direction
        prev_pos = {}     # Tracks last Y for speed
        final_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        direction_counts = {"Inbound": 0, "Outbound": 0}
        
        # Speed Buffers
        speed_data = {"Inbound": [], "Outbound": []}
        
        # Pixels-to-Meters factor (Estimated for typical highway view)
        PXM = 0.06 

        with st.status("Analyzing unique signatures and velocity...", expanded=True) as status:
            results = model.track(source=st.session_state.video_path, stream=True, persist=True, imgsz=320)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.orig_img.copy()
                current_speeds = {"Inbound": [], "Outbound": []}
                
                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confidences):
                        label = model.names[cls]
                        if label in final_counts:
                            y_center = (box[1] + box[3]) / 2
                            
                            # --- SPEED LOGIC ---
                            if obj_id in prev_pos:
                                dy = abs(y_center - prev_pos[obj_id])
                                # Simple Speed: (pixels moved * scaling) / time * 3.6 for km/h
                                speed = (dy * PXM) / (1/fps) * 3.6
                                if speed > 2: # Ignore jitter
                                    if y_center > h/2: current_speeds["Inbound"].append(speed)
                                    else: current_speeds["Outbound"].append(speed)
                            prev_pos[obj_id] = y_center

                            # --- DIRECTION LOGIC ---
                            if obj_id not in entry_points:
                                entry_points[obj_id] = y_center
                                final_counts[label] += 1
                                tracked_ids.add(obj_id)
                            else:
                                if entry_points[obj_id] != 99999 and entry_points[obj_id] != -99999:
                                    if y_center > entry_points[obj_id] + 15: 
                                        direction_counts["Inbound"] += 1
                                        entry_points[obj_id] = 99999 
                                    elif y_center < entry_points[obj_id] - 15:
                                        direction_counts["Outbound"] += 1
                                        entry_points[obj_id] = -99999

                            # Manual Plotting with % Confidence
                            if st.session_state.config['overlays']:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label} {int(conf*100)}%", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store averages for sidebar
                for d in ["Inbound", "Outbound"]:
                    if current_speeds[d]: speed_data[d].append(np.mean(current_speeds[d]))

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
            st.button("🔄 New Analysis החל מחדש", on_click=reset_app)

        # Sidebar Live Status
        if st.session_state.config['sidebar']:
            st.sidebar.title("Live Traffic State")
            
            # Speed-based Status logic
            avg_in = np.mean(speed_data["Inbound"]) if speed_data["Inbound"] else 0
            avg_out = np.mean(speed_data["Outbound"]) if speed_data["Outbound"] else 0
            
            st.sidebar.subheader("Inbound (Down)")
            st.sidebar.markdown(f"**Status:** {get_traffic_state(avg_in)}")
            st.sidebar.metric("Volume", direction_counts["Inbound"])
            
            st.sidebar.divider()
            
            st.sidebar.subheader("Outbound (Up)")
            st.sidebar.markdown(f"**Status:** {get_traffic_state(avg_out)}")
            st.sidebar.metric("Volume", direction_counts["Outbound"])

    except Exception as e:
        st.error(f"System Error: {e}")
        if st.button("Emergency Reset"): reset_app()
