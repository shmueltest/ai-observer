import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip

# --- DARK MODE UI & STYLING ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

# Custom CSS to force a dark, professional aesthetic
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
    st.title("🚦 Traffic Intelligence System")
    st.subheader("Professional Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Upload footage for analysis", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2: BRIEFING ---
elif st.session_state.step == "briefing":
    st.header("📋 Command Briefing")
    
    with st.container(border=True):
        loc = st.text_input("Location Label", "Sector 7G - Main Intersection")
        duration = st.slider("Analysis Window (Seconds)", 1, 60, 10)
        
        c1, c2 = st.columns(2)
        with c1:
            burn_hud = st.toggle("Burn Telemetry into Video", value=True)
            overlays = st.toggle("AI Tracking Boxes", value=True)
        with c2:
            gen_graph = st.toggle("Generate Final Metrics Graph", value=True)
            use_sidebar = st.toggle("Live Sidebar Metrics", value=True)

    if st.button("🚀 INITIATE ANALYSIS"):
        st.session_state.config = {
            "loc": loc, "duration": duration, "burn": burn_hud, 
            "overlays": overlays, "graph": gen_graph, "sidebar": use_sidebar
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3: PROCESSING & UNIQUE TRACKING ---
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

        # --- FIX: UNIQUE ID TRACKING ---
        tracked_ids = set() # Stores unique database IDs
        final_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        
        with st.status("Analyzing unique vehicle signatures...", expanded=True) as status:
            # We use .track with persist=True to maintain IDs across frames
            results = model.track(source=st.session_state.video_path, stream=True, persist=True, imgsz=320)
            
            for i, r in enumerate(results):
                if i >= max_frames: break
                
                frame = r.plot() if st.session_state.config['overlays'] else r.orig_img
                current_v = 0
                
                if r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()
                    
                    for obj_id, cls in zip(ids, clss):
                        label = model.names[cls]
                        if label in final_counts:
                            current_v += 1
                            # Only increment if this ID hasn't been seen before
                            if obj_id not in tracked_ids:
                                final_counts[label] += 1
                                tracked_ids.add(obj_id)

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
            
            # Web Optimization
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
            if st.button("🔄 New Analysis"): reset_app()
            if st.button("🛑 End Session", type="primary"): reset_app()

        if st.session_state.config['sidebar']:
            st.sidebar.title("Live Metrics")
            st.sidebar.info(f"Unique Vehicles: {len(tracked_ids)}")
            for k, v in final_counts.items(): st.sidebar.metric(k.capitalize(), v)

    except Exception as e:
        st.error(f"System Error: {e}")
        if st.button("Emergency Reset"): reset_app()
