import streamlit as st
import cv2
import yt_dlp
import numpy as np
import os
import re
from ultralytics import YOLO

# --- 1. TACTICAL URL INTERCEPTOR ---
def get_secure_url(input_url):
    # Extracts ID and forces the no-cookie domain
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|v/|embed/)|youtu\.be/|youtube-nocookie\.com/embed/)([\w-]{11})'
    match = re.search(regex, input_url)
    if match:
        return f"https://www.youtube-nocookie.com/watch?v={match.group(1)}"
    return input_url

# --- 2. THE BYPASS DOWNLOADER ---
def download_stealth(video_url):
    target_url = get_secure_url(video_url)
    
    ydl_opts = {
        'format': 'best[ext=mp4]', # Merged MP4 is less likely to trigger 403
        'outtmpl': 'input_target.mp4',
        'quiet': True,
        'no_warnings': True,
        # FORCE NODEJS RUNTIME
        'js_runtimes': ['node'],
        'extractor_args': {
            'youtube': {
                # web_embedded is currently the most successful client
                'player_client': ['web_embedded', 'mweb'],
                'player_js_version': 'actual'
            }
        },
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'nocheckcertificate': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([target_url])

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Tactical AI v18.0", layout="wide")
st.title("🛰️ AI Tactical Command Center")

with st.sidebar:
    st.header("📡 Mission Parameters")
    url_input = st.text_input("YouTube Target (URL)", placeholder="Paste link...")
    
    st.divider()
    st.subheader("Manual Backup")
    # If the URL fails, this will always work
    up_file = st.file_uploader("Upload Video (If URL gets 403)", type=['mp4', 'mov'])
    
    st.divider()
    st.subheader("System Toggles")
    conf_boxes = st.checkbox("Draw Bounding Boxes?", value=True)
    conf_sidebar = st.checkbox("Show Tactical Sidebar?", value=True)
    obs_time = st.slider("Observation Time (Seconds)", 5, 60, 15)

# --- 4. EXECUTION ENGINE ---
if st.button("🏁 Initiate Analysis"):
    # Priority: Upload > URL
    if up_file:
        with open("input_target.mp4", "wb") as f:
            f.write(up_file.getbuffer())
        st.sidebar.success("Using Uploaded Intel.")
    elif url_input:
        try:
            with st.status("📡 Establishing Secure Uplink...", expanded=True):
                st.write("🛰️ Solving Security Signatures with Node.js...")
                download_stealth(url_input)
        except Exception as e:
            st.error("❌ YouTube Security Wall detected the Cloud IP (403).")
            st.info("💡 **QUICK FIX:** Download the video to your computer, then use the 'Upload' button in the sidebar.")
            st.stop()
    else:
        st.warning("Input Required.")
        st.stop()

    # --- CV PROCESSING ---
    with st.status("🧠 AI Analyzing Sector...", expanded=True) as status:
        model = YOLO('yolo11n.pt')
        cap = cv2.VideoCapture('input_target.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_w = w * 2 if conf_sidebar else w
        out_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))

        # Filter for vehicles/people
        results = model.track(source='input_target.mp4', stream=True, persist=True, classes=[0,2,3,5,7])
        
        for i, result in enumerate(results):
            if i >= int(fps * obs_time): break
            
            frame = result.plot() if conf_boxes else result.orig_img.copy()
            
            if conf_sidebar:
                # Split screen: Left=Video, Right=Data
                canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
                canvas[:, :w] = frame
                cv2.putText(canvas, "TACTICAL FEED ACTIVE", (w+40, 60), 1, 2, (0, 255, 0), 2)
                cv2.putText(canvas, f"OBJECTS: {len(result.boxes)}", (w+40, 120), 1, 1.5, (255, 255, 255), 2)
                final_frame = canvas
            else:
                final_frame = frame
            
            out_writer.write(final_frame)

        cap.release()
        out_writer.release()
        
        st.write("🎬 Optimizing for Web Playback...")
        os.system("ffmpeg -y -i temp.mp4 -vcodec libx264 -crf 28 output.mp4")
        status.update(label="Mission Complete!", state="complete")

    st.video("output.mp4")
