import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
from ultralytics import YOLO

st.set_page_config(page_title="Tactical AI Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v16.0")

# --- SIDEBAR: REPLACING YES/NO PROMPTS ---
with st.sidebar:
    st.header("📡 Mission Parameters")
    yt_url = st.text_input("YouTube Target (URL)", placeholder="https://...")
    
    st.divider()
    st.subheader("Manual Intel Upload")
    uploaded_file = st.file_uploader("Backup: Upload Video File", type=['mp4', 'mov', 'avi'])
    
    st.divider()
    st.subheader("System Toggles")
    # These replace your old 'input()' prompts
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Sidebar?", value=True)
    conf_traffic = st.checkbox("Enable Flow Intelligence?", value=True)
    
    observation_sec = st.slider("Observation Time (Seconds)", 5, 60, 15)

# --- STEALTH DOWNLOADER ---
def download_stealth(url):
    # Forcing 'ios' and 'mweb' clients often bypasses the 403 error on cloud servers
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'input_target.mp4',
        'quiet': True,
        'extractor_args': {'youtube': {'player_client': ['ios', 'mweb', 'web']}},
        'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- MAIN ENGINE ---
if st.button("🏁 Initiate Analysis"):
    # Priority 1: Use Uploaded File | Priority 2: Use YouTube
    if uploaded_file:
        with open("input_target.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("Using Uploaded Intel.")
    elif yt_url:
        try:
            with st.status("Bypassing YouTube Defense...", expanded=True):
                download_stealth(yt_url)
        except Exception as e:
            st.error(f"YouTube Blocked the Connection (403). Please use the 'Upload Video' option.")
            st.stop()
    else:
        st.warning("Please provide a URL or upload a file.")
        st.stop()

    # --- CV PROCESSING ---
    with st.status("AI Analyzing Sector...", expanded=True) as status:
        model = YOLO('yolo11n.pt')
        cap = cv2.VideoCapture('input_target.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output width depends on sidebar choice
        out_w = w * 2 if conf_sidebar else w
        out_writer = cv2.VideoWriter('raw_render.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

        results = model.track(source='input_target.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
        
        for i, result in enumerate(results):
            if i >= int(fps * observation_sec): break
            
            # Use Sidebar Yes/No logic
            frame = result.plot() if conf_boxes else result.orig_img.copy()
            
            if conf_sidebar:
                canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                canvas[:, :w] = frame
                cv2.putText(canvas, "TACTICAL ANALYSIS ACTIVE", (w+20, 50), 1, 1.5, (0, 255, 255), 2)
                final_frame = canvas
            else:
                final_frame = frame
            
            out_writer.write(final_frame)

        cap.release()
        out_writer.release()
        
        st.write("🎬 Converting for Web Playback...")
        os.system("ffmpeg -y -i raw_render.mp4 -vcodec libx264 -crf 28 output.mp4")
        status.update(label="Analysis Complete!", state="complete")

    st.video("output.mp4")
