import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np
import time

# --- DARK MODE UI ---
st.set_page_config(page_title="Speed Analytics", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #464b5d; }
    .stButton>button:hover { border: 1px solid #ff4b4b; color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def get_speed_status(speed):
    if speed < 10: return "🔴 CONGESTED"
    if speed < 35: return "🟡 HEAVY FLOW"
    return "🟢 FREE FLOW"

if 'step' not in st.session_state:
    st.session_state.step = "welcome"

# --- STEP 1: WELCOME ---
if st.session_state.step == "welcome":
    st.title("🛰️ Speed-Based Traffic Intelligence")
    st.subheader("Velocity Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Upload footage", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2: BRIEFING ---
elif st.session_state.step == "briefing":
    st.header("📋 Analysis Parameters")
    
    with st.container(border=True):
        loc = st.text_input("Location Label", "Highway 101 - Segment A")
        duration = st.slider("Analysis Window (Seconds)", 1, 60, 10)
        
        st.info("ℹ️ Speed estimation uses pixel displacement relative to FPS. Calibration is based on a standard 30m road segment view.")
        
        c1, c2 = st.columns(2)
        with c1:
            burn_hud = st.toggle("Burn Speed Data into Video", value=True)
            overlays = st.toggle("Show AI Boxes", value=True)
        with c2:
            gen_graph = st.toggle("Generate Speed Trend Graph", value=True)
            use_sidebar = st.toggle("Show Live Speed in Sidebar", value=True)

    if st.button("🚀 INITIATE VELOCITY SCAN"):
        st.session_state.config = {
            "loc": loc, "duration": duration, "burn": burn_hud, 
            "overlays": overlays, "graph": gen_graph, "sidebar": use_sidebar
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3: PROCESSING & SPEED TRACKING ---
elif st.session_state.step == "processing":
    st.header("⚙️ Calculating Velocity Profiles...")
    
    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(st.session_state.config['duration'] * fps)

        raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Speed logic variables
        prev_positions = {} # {id: (y_coord, timestamp)}
        speed_history = {"Inbound": [], "Outbound": []}
        
        # Calibration constant: pixels to meters (approximate for highway view)
        # In a real app, this would be set by user calibration
        PXM = 0.05 

        with st.status("Computing vectors...", expanded=True) as status:
            results = model.track(source=st.session_state.video_path, stream=True, persist=True, imgsz=320)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.orig_img.copy()
                current_frame_speeds = {"Inbound": [], "Outbound": []}
                
                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confidences):
                        label = model.names[cls]
                        if label in ["car", "truck", "bus"]:
                            y_center = (box[1] + box[3]) / 2
                            
                            # Speed Calculation
                            if obj_id in prev_positions:
                                prev_y, prev_time = prev_positions[obj_id]
                                dy = abs(y_center - prev_y)
                                dt = 1/fps
                                # Calculate speed: (dist in px * PXM) / time * 3.6 for km/h
                                velocity = (dy * PXM) / dt * 3.6
                                
                                if y_center > h/2:
                                    current_frame_speeds["Inbound"].append(velocity)
                                    speed_history["Inbound"].append(velocity)
                                else:
                                    current_frame_speeds["Outbound"].append(velocity)
                                    speed_history["Outbound"].append(velocity)
                            
                            prev_positions[obj_id] = (y_center, i/fps)

                            if st.session_state.config['overlays']:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                # Labels showing % confidence
                                cv2.putText(frame, f"{label} {int(conf*100)}%", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- HUD RENDERING ---
                if st.session_state.config['burn']:
                    avg_in = np.mean(current_frame_speeds["Inbound"]) if current_frame_speeds["Inbound"] else 0
                    avg_out = np.mean(current_frame_speeds["Outbound"]) if current_frame_speeds["Outbound"] else 0
                    
                    sidebar_w = int(w * 0.3)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, "VELOCITY TELEMETRY", (w - sidebar_w + 20, 50), 2, 0.7, (255, 255, 0), 1)
                    cv2.putText(frame, f"INBOUND: {int(avg_in)} km/h", (w - sidebar_w + 20, 120), 2, 0.6, (255,255,255), 1)
                    cv2.putText(frame, f"STATUS: {get_speed_status(avg_in)}", (w - sidebar_w + 20, 160), 2, 0.5, (0,255,0), 1)
                    
                    cv2.putText(frame, f"OUTBOUND: {int(avg_out)} km/h", (w - sidebar_w + 20, 240), 2, 0.6, (255,255,255), 1)
                    cv2.putText(frame, f"STATUS: {get_speed_status(avg_out)}", (w - sidebar_w + 20, 280), 2, 0.5, (0,255,0), 1)

                out.write(cv2.resize(frame, (w, h)))
            
            out.release()
            cap.release()
            
            # Export
            final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            clip = VideoFileClip(raw_path)
            clip.write_videofile(final_path, codec="libx264", audio=False)
            status.update(label="Velocity Profile Generated", state="complete")

        # --- RESULTS ---
        st.video(final_path)
        
        if st.session_state.config['graph']:
            st.write("### 📈 Speed Distribution")
            all_speeds = speed_history["Inbound"] + speed_history["Outbound"]
            if all_speeds:
                df_speed = pd.DataFrame(all_speeds, columns=['km/h'])
                st.line_chart(df_speed)

        st.button("🔄 Restart Analyst", on_click=reset_app)

    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Reset"): reset_app()
