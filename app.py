import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
import re
from ultralytics import YOLO

# --- SECURE DOWNLOADER ---
def download_video_failproof(url):
    # Sanitize URL
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|v/|embed/)|youtu\.be/|youtube-nocookie\.com/embed/)([\w-]{11})'
    match = re.search(regex, url)
    target = f"https://www.youtube-nocookie.com/watch?v={match.group(1)}" if match else url

    ydl_opts = {
        'format': 'best[ext=mp4]', 
        'outtmpl': 'input_target.mp4',
        'quiet': True,
        'js_runtimes': ['node'],
        'extractor_args': {
            'youtube': {
                # 2026 Meta: 'ios' and 'mweb' are the most resilient
                'player_client': ['ios', 'mweb', 'web_embedded'],
                'player_js_version': 'actual'
            }
        },
        'impersonate': 'chrome', # Uses curl-cffi to mimic Chrome 
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([target])

# --- UI ---
st.set_page_config(page_title="Tactical AI v19", layout="wide")
st.title("🛰️ AI Tactical Command Center")

with st.sidebar:
    st.header("📡 Mission Parameters")
    yt_url = st.text_input("YouTube URL")
    
    st.divider()
    st.subheader("🛠️ 403 Emergency Bypass")
    st.info("If the URL fails, YouTube has blocked this server's IP. Upload the video file here instead:")
    up_file = st.file_uploader("Manual Video Upload", type=['mp4', 'mov', 'avi'])
    
    st.divider()
    conf_boxes = st.checkbox("Bounding Boxes", value=True)
    obs_time = st.slider("Seconds to Analyze", 5, 60, 10)

# --- ENGINE ---
if st.button("🏁 Initiate Analysis"):
    source_file = None
    
    if up_file:
        with open("input_target.mp4", "wb") as f:
            f.write(up_file.getbuffer())
        source_file = "input_target.mp4"
    elif yt_url:
        try:
            with st.status("📡 Attempting Secure Bypass..."):
                download_video_failproof(yt_url)
                source_file = "input_target.mp4"
        except Exception:
            st.error("❌ 403 Forbidden: YouTube's Security is blocking this Cloud Server.")
            st.warning("👉 **FIX:** Download the video to your computer first, then use the 'Manual Video Upload' button in the sidebar.")
            st.stop()
    
    if source_file:
        with st.status("🧠 AI Processing...", expanded=True) as status:
            model = YOLO('yolo11n.pt')
            cap = cv2.VideoCapture(source_file)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            results = model.track(source=source_file, stream=True, persist=True, classes=[0,2,3,5,7])
            
            for i, result in enumerate(results):
                if i >= int(fps * obs_time): break
                frame = result.plot() if conf_boxes else result.orig_img
                out_writer.write(frame)
            
            cap.release()
            out_writer.release()
            
            os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Analysis Complete!", state="complete")
        
        st.video("output.mp4")
