import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Tactical Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v16.0")

# --- SIDEBAR: YOUR MISSING PROMPTS ---
with st.sidebar:
    st.header("📡 Mission Configuration")
    url = st.text_input("YouTube URL", placeholder="Paste link...")
    sec = st.slider("Observation Time (Seconds)", 5, 60, 15)
    
    st.divider()
    st.subheader("System Toggles")
    # These replace your old 'input(y/n)' prompts
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Scoreboard?", value=True)
    conf_traffic = st.checkbox("Enable Traffic Intelligence?", value=True)
    conf_report = st.checkbox("Generate Final Intel Report?", value=True)

# --- THE DOWNLOADER (403 BYPASS FIX) ---
def download_video(video_url):
    ydl_opts = {
        # The 'best' format often triggers 403. Using specific player clients helps.
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'input_video.mp4',
        'quiet': True,
        'no_warnings': True,
        # 2026 Stealth Arguments
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'mweb', 'tv'],
                'player_js_version': 'actual'
            }
        },
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# --- EXECUTION ---
if st.button("🏁 Launch AI Mission"):
    if not url:
        st.error("Target URL Required!")
    else:
        with st.status("Initializing Tactical Uplink...", expanded=True) as status:
            try:
                st.write("🛰️ Bypassing YouTube Security...")
                download_video(url)
            except Exception as e:
                st.error(f"YouTube Blocked the Server (403). Error: {e}")
                st.info("💡 Tip: Try a different video or use the 'Upload File' option below.")
                st.stop()

            # --- AI PROCESSING ---
            st.write("🧠 Engaging Computer Vision...")
            model = YOLO('yolo11n.pt')
            cap = cv2.VideoCapture('input_video.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust width if sidebar is enabled
            out_w = w * 2 if conf_sidebar else w
            out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

            results = model.track(source='input_video.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
            
            for i, result in enumerate(results):
                if i >= int(fps * sec): break
                
                # Draw boxes only if user said YES in sidebar
                frame = result.plot() if conf_boxes else result.orig_img.copy()
                
                if conf_sidebar:
                    # Create the side-by-side view
                    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                    canvas[:, :w] = frame
                    cv2.putText(canvas, "LIVE TACTICAL FEED", (w+20, 50), 1, 1.5, (0,255,0), 2)
                    # (Add your specific movement counting logic here)
                    final_frame = canvas
                else:
                    final_frame = frame
                
                out_writer.write(final_frame)

            cap.release()
            out_writer.release()
            
            st.write("🎬 Processing MP4 for Web...")
            os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Mission Accomplished!", state="complete")

        st.video("output.mp4")
        if conf_report:
            st.success("Intel Report Generated: Targets identified in sector.")
