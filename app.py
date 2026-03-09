import streamlit as st
import cv2
import yt_dlp
import datetime
import numpy as np
import os
from ultralytics import YOLO

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Tactical Observer", layout="wide")
st.title("🛰️ AI Tactical Command Center v15.5")

# --- ENGINE CLASS ---
class StreamlitObserverAI:
    def __init__(self):
        # Cache model for speed
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
        
        # Anti-Plant Filter (Displacement Check)
        total_dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
        if total_dist < 45: return None 

        dx, dy = cx - px, cy - py
        if abs(dy) > abs(dx): return "North" if dy < 0 else "South"
        else: return "West" if dx < 0 else "East"

    def draw_ui(self, frame, config):
        h, w, _ = frame.shape
        if config['scoreboard']:
            canvas_w = w * 2
            canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
            canvas[:, :w] = frame
            sb_start = w
            cv2.rectangle(canvas, (sb_start, 0), (canvas_w, h), (15, 15, 15), -1)
            cv2.line(canvas, (sb_start, 0), (sb_start, h), (0, 255, 255), 2)
            
            # Text Summary
            cv2.putText(canvas, "TACTICAL SUMMARY", (sb_start + 30, 50), 1, 1.2, (0, 255, 255), 2)
            y_off = 100
            for d in self.directions:
                total = sum(self.data[d].values())
                cv2.putText(canvas, f"{d}: {total}", (sb_start + 30, y_off), 1, 1.1, (0, 255, 0), 2)
                y_off += 50
            return canvas
        return frame

# --- SIDEBAR CONFIGURATION (YOUR PROMPTS) ---
with st.sidebar:
    st.header("📡 Command Inputs")
    url = st.text_input("YouTube URL", placeholder="Paste link here...")
    sec = st.slider("Observation Time (Seconds)", 5, 60, 15)
    
    st.divider()
    st.write("🔧 System Toggles")
    config = {
        'boxes': st.checkbox("Draw Tracking Boxes", value=True),
        'scoreboard': st.checkbox("Enable Detailed Sidebar", value=True),
        'report': st.checkbox("Generate Final Report", value=True),
        'traffic_logic': st.checkbox("Enable Traffic Congestion Alerts", value=True)
    }
    
    st.warning("Cloud processing is CPU-based and may take a moment.")

# --- EXECUTION ENGINE ---
if st.button("🏁 Launch AI Mission"):
    if not url:
        st.error("Missing Target URL!")
    else:
        engine = StreamlitObserverAI()
        
        with st.status("Initializing Tactical Uplink...", expanded=True) as status:
            # FIX FOR DOWNLOAD ERROR: Use specific format and quiet mode
            st.write("🛰️ Bypassing security & downloading stream...")
            ydl_opts = {
                'format': 'best[ext=mp4]/best', # Force MP4
                'outtmpl': 'input.mp4',
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                st.error(f"YouTube Blocked the Connection: {e}")
                st.stop()
            
            # YOLO Processing
            st.write("🧠 Processing Visual Intelligence...")
            cap = cv2.VideoCapture('input.mp4')
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust output width based on sidebar toggle
            out_w = w * 2 if config['scoreboard'] else w
            out_writer = cv2.VideoWriter('temp_render.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))
            
            # Class list (no plants): person, car, motorcycle, bus, truck
            results = engine.model.track(source='input.mp4', stream=True, persist=True, conf=0.3, classes=[0,2,3,5,7])
            
            progress = st.progress(0)
            for i, result in enumerate(results):
                if i >= int(fps * sec): break
                
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

                frame = result.plot() if config['boxes'] else result.orig_img.copy()
                final_frame = engine.draw_ui(frame, config)
                out_writer.write(final_frame)
                progress.progress((i + 1) / int(fps * sec))

            cap.release()
            out_writer.release()
            
            st.write("🎞️ Finalizing Video...")
            os.system("ffmpeg -y -i temp_render.mp4 -vcodec libx264 -crf 28 output.mp4")
            status.update(label="Mission Accomplished!", state="complete")

        st.video("output.mp4")
        
        if config['report']:
            st.subheader("📋 Mission Report")
            st.write(engine.data)
