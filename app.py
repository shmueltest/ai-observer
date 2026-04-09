
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np

# --- UI ---
st.set_page_config(page_title="Traffic Tracker", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #445b8e; color: #a10303 }
.stButton>button { width: 100%; height: 3em; background-color: #262730; color: white; }
</style>
""", unsafe_allow_html=True)

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def get_traffic_state(avg_speed):
    if avg_speed == 0: return "NO DATA", (200,200,200)
    if avg_speed < 15: return "HEAVY", (0,0,255)
    if avg_speed < 40: return "MODERATE", (0,255,255)
    return "LIGHT", (0,255,0)

if 'step' not in st.session_state:
    st.session_state.step = "welcome"
if 'config' not in st.session_state:
    st.session_state.config = {}

# --- STEP 1 ---
if st.session_state.step == "welcome":
    st.title("🚦 Traffic Tracking System")

    uploaded_file = st.file_uploader("Upload footage", type=['mp4','mov','avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# --- STEP 2 ---
elif st.session_state.step == "briefing":
    st.header("📋 Analysis Briefing")

    loc = st.text_input("Location", "Main Intersection")
    duration = st.slider("Seconds", 1, 60, 10)

    burn = st.toggle("Burn HUD", True)
    overlays = st.toggle("Boxes", True)
    graph = st.toggle("Graph", True)
    sidebar = st.toggle("Sidebar", True)

    if st.button("🚀 Start"):
        st.session_state.config = {
            "loc": loc, "duration": duration,
            "burn": burn, "overlays": overlays,
            "graph": graph, "sidebar": sidebar
        }
        st.session_state.step = "processing"
        st.rerun()

# --- STEP 3 ---
elif st.session_state.step == "processing":
    st.header("🧠 Processing...")

    try:
        model = YOLO("yolo11n.pt")
        cap = cv2.VideoCapture(st.session_state.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        max_frames = int(st.session_state.config['duration'] * fps)

        raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

        # --- TRACKING ---
        LINE_Y = int(h * 0.5)
        OFFSET = 10

        counted_ids = set()
        tracked_ids = set()
        prev_pos = {}

        final_counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}
        direction_counts = {"Inbound":0,"Outbound":0}
        speed_history = {"Inbound":[], "Outbound":[]}

        PXM = 0.06

        with st.status("Analyzing...", expanded=True):

            results = model.track(
                source=st.session_state.video_path,
                stream=True,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=320
            )

            for i, r in enumerate(results):
                if i >= max_frames:
                    break

                frame = r.orig_img.copy()
                current_speeds = {"Inbound":[], "Outbound":[]}

                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    clss = r.boxes.cls.int().cpu().tolist()
                    confs = r.boxes.conf.cpu().numpy()

                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
                        label = model.names[cls]

                        if label not in final_counts:
                            continue
                        if conf < 0.4:
                            continue

                        x1,y1,x2,y2 = box
                        y_center = (y1+y2)/2

                        area = (x2-x1)*(y2-y1)
                        if area < 500:
                            continue

                        tracked_ids.add(obj_id)

                        # --- SPEED ---
                        if obj_id in prev_pos:
                            dy = abs(y_center - prev_pos[obj_id])
                            speed = (dy * PXM) / (1/fps) * 3.6

                            if speed > 2:
                                if y_center > LINE_Y:
                                    current_speeds["Inbound"].append(speed)
                                else:
                                    current_speeds["Outbound"].append(speed)

                        # --- COUNTING ---
                        if obj_id in prev_pos:
                            prev_y = prev_pos[obj_id]

                            if obj_id not in counted_ids:

                                if prev_y < LINE_Y - OFFSET and y_center >= LINE_Y + OFFSET:
                                    final_counts[label] += 1
                                    direction_counts["Inbound"] += 1
                                    counted_ids.add(obj_id)

                                elif prev_y > LINE_Y + OFFSET and y_center <= LINE_Y - OFFSET:
                                    final_counts[label] += 1
                                    direction_counts["Outbound"] += 1
                                    counted_ids.add(obj_id)

                        prev_pos[obj_id] = y_center

                        # --- DRAW BOX ---
                        if st.session_state.config['overlays']:
                            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                            cv2.putText(frame,f"{label} {int(conf*100)}%",
                                (int(x1),int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                # --- LINE ---
                cv2.line(frame,(0,LINE_Y),(w,LINE_Y),(0,0,255),2)

                # --- SPEED HISTORY ---
                for d in ["Inbound","Outbound"]:
                    if current_speeds[d]:
                        speed_history[d].append(np.mean(current_speeds[d]))

                # --- HUD ---
                if st.session_state.config['burn']:
                    sidebar_w = int(w*0.3)
                    overlay = frame.copy()
                    cv2.rectangle(overlay,(w-sidebar_w,0),(w,h),(0,0,0),-1)
                    cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)

                    cv2.putText(frame, st.session_state.config['loc'].upper(),
                                (w-sidebar_w+20,40),1,1.2,(255,255,255),2)

                    y_off = 100
                    for obj,val in final_counts.items():
                        cv2.putText(frame,f"{obj.upper()}: {val}",
                                    (w-sidebar_w+20,y_off),1,1,(200,200,200),1)
                        y_off += 30

                    avg_in = speed_history["Inbound"][-1] if speed_history["Inbound"] else 0
                    avg_out = speed_history["Outbound"][-1] if speed_history["Outbound"] else 0

                    state_in,color_in = get_traffic_state(avg_in)
                    state_out,color_out = get_traffic_state(avg_out)

                    y_off += 40
                    cv2.putText(frame,"TRAFFIC STATUS",(w-sidebar_w+20,y_off),1,1,(255,255,0),2)
                    y_off += 40
                    cv2.putText(frame,f"IN: {state_in}",(w-sidebar_w+20,y_off),1,1,color_in,2)
                    y_off += 40
                    cv2.putText(frame,f"OUT: {state_out}",(w-sidebar_w+20,y_off),1,1,color_out,2)

                out.write(cv2.resize(frame,(w,h)))

        out.release()
        cap.release()

        final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        clip = VideoFileClip(raw_path)
        clip.write_videofile(final_path, codec="libx264", audio=False)

        # --- RESULTS ---
        st.video(final_path)

        with open(final_path,'rb') as f:
            st.download_button("📥 Download", f.read(), "report.mp4")

        if st.session_state.config['graph']:
            df = pd.DataFrame(list(final_counts.items()), columns=['Vehicle','Total'])
            st.bar_chart(df.set_index('Vehicle'))

        if st.session_state.config['sidebar']:
            st.sidebar.metric("Unique IDs", len(tracked_ids))
            st.sidebar.metric("Inbound", direction_counts["Inbound"])
            st.sidebar.metric("Outbound", direction_counts["Outbound"])

        st.button("🔄 Reset", on_click=reset_app)

    except Exception as e:
        st.error(f"Error: {e}")
                        
