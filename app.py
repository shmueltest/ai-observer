import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Tactical Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v16.5")

# --- SIDEBAR: TACTICAL CONFIGURATION (REPLACES YES/NO PROMPTS) ---
with st.sidebar:
    st.header("📡 Mission Parameters")
    url = st.text_input("YouTube Target (URL)", placeholder="Paste link and press Enter...")
    
    st.divider()
    st.subheader("System Toggles")
    # These replace your old 'input("Yes/No")' calls
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Scoreboard?", value=True)
    conf_traffic = st.checkbox("Enable Flow Intelligence?", value=True)
    
    obs_time = st.slider("Observation Time (Seconds)", 5, 60, 15)
    st.info("Note: First run takes longer to load AI weights.")

# --- STEALTH DOWNLOADER (THE BYPASS) ---
def download_stealth(video_url):
    ydl_opts = {
        # 'best' is often blocked. Asking for specific mp4 is safer.
        'format': 'best[ext=mp4]', 
        'outtmpl': 'input_video.mp4',
        'quiet': True,
        'no_warnings': True,
        # 2026 BYPASS: Use 'ios' and 'android_sdkless' to trick YouTube
        'extractor_args': {
            'youtube': {
                'player_client': ['ios', 'web', 'mweb', '-android_sdkless'],
                'player_js_version': 'actual'
            }
        },
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# --- EXECUTION ENGINE ---
if st.button("🏁 Initiate Tactical Analysis"):
    if not url:
        st.error("Target URL Required! Please paste a YouTube link.")
    else:
        with st.status("📡 Establishing Uplink...", expanded=True) as status:
            try:
                st.write("🛰️ Attempting Security Bypass...")
                download_stealth(url)
            except Exception as e:
                st.error(f"YouTube Blocked the Server (403).")
                st.write("TIP: Try a different video or use a 'no-cookie' link variant.")
                st.stop()

            # --- AI PROCESSING ---
            st.write("🧠 Engaging Computer Vision (YOLO)...")
            model = YOLO('yolo11n.pt')
            cap = cv2.VideoCapture('input_video.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output canvas based on your YES/NO choice
            out_w = w * 2 if conf_sidebar else w
            out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

            # Run tracking
            results = model.track(source='input_video.mp4', stream=True, persist=True, conf=0.3, classes=[0,2,3,5,7])
            
            progress = st.progress(0)
            for i, result in enumerate(results):
                if i >= int(fps * obs_time): break
                
                # Render based on your toggles
                frame = result.plot() if conf_boxes else result.orig_img.copy()
                
                if conf_sidebar:
                    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                    canvas[:, :w] = frame
                    cv2.putText(canvas, "TACTICAL FEED ACTIVE", (w+20, 50), 1, 1.5, (0, 255, 0), 2)
                    # You can add more movement logic here
                    final_frame = canvas
                else:
                    final_frame = frame
                
                out_writer.write(final_frame)
                progress.progress((i + 1) / int(fps * obs_time))

            cap.release()
            out_writer.release()
            
            st.write("🎬 Finalizing Video Intelligence...")
            os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Mission Accomplished!", state="complete")

        st.video("output.mp4")
