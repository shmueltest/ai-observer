import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
import re
from ultralytics import YOLO

# --- 1. TACTICAL URL FIXER ---
def get_nocookie_url(input_url):
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|v/|embed/)|youtu\.be/|youtube-nocookie\.com/embed/)([\w-]{11})'
    match = re.search(regex, input_url)
    if match:
        return f"https://www.youtube-nocookie.com/watch?v={match.group(1)}"
    return input_url

# --- 2. THE BYPASS DOWNLOADER ---
def download_video(video_url):
    target_url = get_nocookie_url(video_url)
    
def download_video(url):
    ydl_opts = {
        # 'best' is often blocked for separate streams. Use merged mp4.
        'format': 'best[ext=mp4]', 
        'outtmpl': 'input_video.mp4',
        'quiet': True,
        'extractor_args': {
            'youtube': {
                # web_embedded is the 2026 "Secret Sauce"
                'player_client': ['web_embedded'],
                'player_js_version': 'actual'
            }
        },
        # Identify as a real browser
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'nocheckcertificate': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Tactical AI v17.0", layout="wide")
st.title("🛰️ AI Tactical Command Center")

with st.sidebar:
    st.header("📡 Mission Parameters")
    yt_link = st.text_input("YouTube URL", placeholder="Paste link here...")
    
    st.divider()
    st.subheader("Manual Backup")
    # Failsafe if YouTube is being stubborn
    up_file = st.file_uploader("Upload Video (If URL gets 403)", type=['mp4', 'mov'])
    
    st.divider()
    st.subheader("System Toggles")
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Scoreboard?", value=True)
    obs_time = st.slider("Observation Time (Seconds)", 5, 60, 15)

# --- 4. MAIN ENGINE ---
if st.button("🏁 Initiate Analysis"):
    # Decide source
    if up_file:
        with open("input_target.mp4", "wb") as f:
            f.write(up_file.getbuffer())
        st.sidebar.success("Using Uploaded File.")
    elif yt_link:
        try:
            with st.status("📡 Bypassing YouTube Security...", expanded=True):
                download_video(yt_link)
        except Exception as e:
            st.error("❌ YouTube Blocked the Cloud Server (403).")
            st.info("💡 Solution: Download the video to your Mac and use the 'Upload' button above.")
            st.stop()
    else:
        st.warning("Please provide a URL or upload a file.")
        st.stop()

    # --- CV PROCESSING ---
    with st.status("🧠 AI Analyzing Sector...", expanded=True) as status:
        model = YOLO('yolo11n.pt')
        cap = cv2.VideoCapture('input_target.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_w = w * 2 if conf_sidebar else w
        out_writer = cv2.VideoWriter('raw_render.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

        results = model.track(source='input_target.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
        
        for i, result in enumerate(results):
            if i >= int(fps * obs_time): break
            
            # Sidebar logic
            frame = result.plot() if conf_boxes else result.orig_img.copy()
            
            if conf_sidebar:
                canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                canvas[:, :w] = frame
                cv2.putText(canvas, "TACTICAL ANALYSIS ACTIVE", (w+20, 50), 1, 1.5, (0, 255, 0), 2)
                # Count logic
                counts = len(result.boxes)
                cv2.putText(canvas, f"TARGETS: {counts}", (w+20, 100), 1, 1.2, (255, 255, 255), 2)
                final_frame = canvas
            else:
                final_frame = frame
            
            out_writer.write(final_frame)

        cap.release()
        out_writer.release()
        
        st.write("🎬 Converting for Playback...")
        os.system("ffmpeg -y -i raw_render.mp4 -vcodec libx264 -crf 28 output.mp4")
        status.update(label="Analysis Complete!", state="complete")

    st.video("output.mp4")
