import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
import re
from ultralytics import YOLO

# --- BYPASS HELPERS ---
def get_nocookie_url(input_url):
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|v/|embed/)|youtu\.be/|youtube-nocookie\.com/embed/)([\w-]{11})'
    match = re.search(regex, input_url)
    return f"https://www.youtube-nocookie.com/watch?v={match.group(1)}" if match else input_url

def download_video(url):
    target_url = get_nocookie_url(url)
    ydl_opts = {
        # 2026 Strategy: Use web_embedded and force no-android-sdkless
        'format': 'best[ext=mp4]', 
        'outtmpl': 'input_target.mp4',
        'quiet': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['web_embedded', 'ios'],
                'player_js_version': 'actual'
            }
        },
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'nocheckcertificate': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([target_url])

# --- UI SETUP ---
st.set_page_config(page_title="Tactical AI v18.0", layout="wide")
st.title("🛰️ AI Tactical Command Center")

with st.sidebar:
    st.header("📡 Mission Parameters")
    yt_link = st.text_input("Paste YouTube Link", placeholder="https://...")
    
    st.divider()
    st.subheader("⚠️ 403 FAILSAFE")
    # This is your insurance policy. If the URL fails, upload the file here.
    up_file = st.file_uploader("Upload Video File Directly", type=['mp4', 'mov'])
    
    st.divider()
    st.subheader("AI Configuration")
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Scoreboard?", value=True)
    obs_time = st.slider("Analyze First X Seconds", 5, 60, 15)

# --- EXECUTION ---
if st.button("🏁 Launch AI Mission"):
    # 1. Determine Source
    if up_file:
        with open("input_target.mp4", "wb") as f:
            f.write(up_file.getbuffer())
        st.success("Analysis starting from uploaded file...")
    elif yt_link:
        try:
            with st.status("📡 Bypassing YouTube Security...", expanded=True):
                download_video(yt_link)
        except Exception:
            st.error("❌ YouTube Blocked the Server (403).")
            st.info("💡 **How to fix this:** YouTube blocks cloud IPs. Download the video to your Mac, then use the 'Upload' button in the sidebar to run the AI.")
            st.stop()
    else:
        st.warning("Please provide a link or upload a file.")
        st.stop()

    # 2. AI & CV ENGINE
    with st.status("🧠 AI Processing Sector...", expanded=True) as status:
        model = YOLO('yolo11n.pt')
        cap = cv2.VideoCapture('input_target.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_w = w * 2 if conf_sidebar else w
        out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

        results = model.track(source='input_target.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
        
        for i, result in enumerate(results):
            if i >= int(fps * obs_time): break
            
            frame = result.plot() if conf_boxes else result.orig_img.copy()
            
            if conf_sidebar:
                canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                canvas[:, :w] = frame
                # Add tactical overlay
                cv2.putText(canvas, f"DETECTIONS: {len(result.boxes)}", (w+40, 100), 1, 1.5, (0, 255, 0), 2)
                final_frame = canvas
            else:
                final_frame = frame
            
            out_writer.write(final_frame)

        cap.release()
        out_writer.release()
        
        st.write("🎬 Finalizing Video for Web...")
        os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
        status.update(label="Mission Complete!", state="complete")

    st.video("output.mp4")
