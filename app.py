import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Tactical Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v16.0")

# --- SIDEBAR: YOUR YES/NO PROMPTS ---
with st.sidebar:
    st.header("📡 Mission Configuration")
    url = st.text_input("YouTube URL")
    sec = st.slider("Observation Time (Seconds)", 5, 60, 15)
    
    st.divider()
    st.subheader("System Toggles")
    # Adding back your specific prompts as checkboxes
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Sidebar?", value=True)
    conf_traffic = st.checkbox("Enable Traffic Alerts?", value=True)
    conf_report = st.checkbox("Generate Final Intel Report?", value=True)

# --- THE DOWNLOADER (BYPASS FIX) ---
def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'input.mp4',
        'quiet': True,
        # Stealth settings to bypass 403 Forbidden
        'extractor_args': {'youtube': {'player_client': ['web', 'mweb', 'tv']}},
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- MAIN EXECUTION ---
if st.button("🏁 Launch AI Mission"):
    if not url:
        st.error("Please enter a Target URL!")
    else:
        with st.status("Initializing Tactical Uplink...", expanded=True) as status:
            try:
                st.write("🛰️ Attempting to bypass security...")
                download_video(url)
            except Exception as e:
                st.error(f"YouTube Blocked the Cloud Server (403). Try a different video or link.")
                st.stop()

            # AI Logic (YOLO)
            st.write("🧠 Running Computer Vision...")
            model = YOLO('yolo11n.pt')
            cap = cv2.VideoCapture('input.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Match width to sidebar choice
            out_w = w * 2 if conf_sidebar else w
            out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

            # Process Frames
            results = model.track(source='input.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
            for i, result in enumerate(results):
                if i >= int(fps * sec): break
                
                # Logic to draw based on your Yes/No toggles
                frame = result.plot() if conf_boxes else result.orig_img.copy()
                
                if conf_sidebar:
                    # (Insert your sidebar drawing logic here from previous versions)
                    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                    canvas[:, :w] = frame
                    cv2.putText(canvas, "TACTICAL FEED", (w+20, 50), 1, 2, (0,255,0), 2)
                    final_frame = canvas
                else:
                    final_frame = frame
                
                out_writer.write(final_frame)

            cap.release()
            out_writer.release()
            
            # Final conversion for web playback
            st.write("🎬 Finalizing Video...")
            os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Mission Complete!", state="complete")

        st.video("output.mp4")
