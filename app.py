import streamlit as st
import cv2
import yt_dlp
import datetime
import numpy as np
import os
from ultralytics import YOLO
import tempfile

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Tactical Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v15.0")
st.subheader("Real-time Traffic & Movement Intelligence")

# --- ENGINE CLASS ---
class StreamlitObserverAI:
    def __init__(self):
        # Cache the model so it doesn't reload on every button click
        self.model = YOLO('yolo11n.pt') 
        self.directions = ["North", "South", "East", "West"]
        self.reset_data()

    def reset_data(self):
        self.counted_ids = {} 
        self.prev_positions = {} 
        self.start_positions = {} 
        self.data = {d: {} for d in self.directions} 

    def get_direction(self, id, current_pos):
        if id not in self.prev_positions:
            self.prev_positions[id] = current_pos
            self.start_positions[id] = current_pos
            return None
        px, py = self.prev_positions[id]; cx, cy = current_pos
        sx, sy = self.start_positions[id]
        
        # Displacement Filter (Anti-Plant)
        total_dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
        if total_dist < 45: return None 

        dx, dy = cx - px, cy - py
        if abs(dy) > abs(dx): return "North" if dy < 0 else "South"
        else: return "West" if dx < 0 else "East"

    def draw_sidebar(self, frame, obj_count):
        h, w, _ = frame.shape
        canvas_w = w * 2
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame # Left side video
        
        sb_start = w
        cv2.rectangle(canvas, (sb_start, 0), (canvas_w, h), (15, 15, 15), -1)
        cv2.line(canvas, (sb_start, 0), (sb_start, h), (0, 255, 255), 2)

        # Intelligence Narrative
        cv2.putText(canvas, "LIVE MISSION DESCRIPTION", (sb_start + 30, 50), 1, 1.2, (0, 255, 255), 2)
        y_nar = 100
        active_found = False
        for d in self.directions:
            d_count = sum(self.data[d].values())
            if d_count > 0:
                active_found = True
                msg = f"HEAVY {d}bound congestion." if d_count > 10 else f"Active flow {d}bound."
                cv2.putText(canvas, f">> {msg}", (sb_start + 30, y_nar), 1, 0.9, (200, 200, 200), 1)
                y_nar += 35
        if not active_found:
            cv2.putText(canvas, ">> Scanning environment...", (sb_start + 30, y_nar), 1, 0.9, (150, 150, 150), 1)

        # Object Counts
        y_cnt = h // 2
        cv2.putText(canvas, "LIVE OBJECT COUNTS", (sb_start + 30, y_cnt), 1, 1.2, (255, 255, 255), 2)
        y_cnt += 50
        for d in self.directions:
            total = sum(self.data[d].values())
            cv2.putText(canvas, f"{d.upper()}: {total}", (sb_start + 30, y_cnt), 1, 1.1, (0, 255, 0), 2)
            y_cnt += 50
        return canvas

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    sec = st.slider("Seconds to Analyze", 5, 60, 15)
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    st.info("Note: Processing on CPU takes ~2-3x the video length.")

# --- EXECUTION ---
if st.button("🚀 Initialize AI Observation"):
    if not url:
        st.error("Please enter a YouTube URL first!")
    else:
        engine = StreamlitObserverAI()
        
        with st.status("Downloading & Analyzing...", expanded=True) as status:
            # 1. Download
            st.write("📥 Fetching video stream...")
            ydl_opts = {'format': 'mp4', 'outtmpl': 'input.mp4', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
            
            # 2. Process
            st.write("🧠 Running YOLO Computer Vision...")
            cap = cv2.VideoCapture('input.mp4')
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Using a temporary file for the output to avoid permission issues
            tmp_out = "temp_render.mp4"
            out_writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
            
            # Track classes: person, car, motorcycle, bus, truck
            results = engine.model.track(source='input.mp4', stream=True, persist=True, conf=0.35, classes=[0,2,3,5,7])
            
            progress_bar = st.progress(0)
            max_frames = int(fps * sec)
            
            for i, result in enumerate(results):
                if i >= max_frames: break
                
                current_objs = 0
                if result.boxes is not None and result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.int().cpu().tolist()
                    
                    for obj_id, box, cls_idx in zip(ids, boxes, classes):
                        label = result.names[cls_idx]
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        direction = engine.get_direction(obj_id, (cx, cy))
                        
                        if direction and obj_id not in engine.counted_ids:
                            engine.counted_ids[obj_id] = direction
                            engine.data[direction][label] = engine.data[direction].get(label, 0) + 1
                    current_objs = len(ids)

                frame = result.plot() if show_boxes else result.orig_img.copy()
                final_frame = engine.draw_sidebar(frame, current_objs)
                out_writer.write(final_frame)
                progress_bar.progress((i + 1) / max_frames)

            cap.release()
            out_writer.release()
            
            # 3. Transcode for Web (H.264)
            st.write("🎬 Finalizing Video Format...")
            os.system(f"ffmpeg -y -i {tmp_out} -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # 4. Display Results
        st.video("output.mp4")
        st.balloons()
