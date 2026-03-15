import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(page_title="Traffic AI System", layout="wide")

# -------------------------------------------------------
# UI STYLE
# -------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B0F14;
    color: white;
}

.stButton>button {
    width:100%;
    height:3em;
    font-size:16px;
    border-radius:8px;
    background-color:#1F2937;
    color:white;
}

[data-testid="stMetric"]{
    background-color:#111827;
    padding:15px;
    border-radius:10px;
    border:1px solid #374151;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# RESET
# -------------------------------------------------------

def reset_app():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# -------------------------------------------------------
# TRAFFIC STATE
# -------------------------------------------------------

def get_traffic_state(speed):

    if speed == 0:
        return "NO DATA / אין נתונים",(200,200,200)

    if speed < 15:
        return "HEAVY / עומס כבד",(0,0,255)

    if speed < 40:
        return "MODERATE / עומס בינוני",(0,255,255)

    return "LIGHT / תנועה קלה",(0,255,0)

# -------------------------------------------------------
# SESSION
# -------------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step="welcome"

# -------------------------------------------------------
# STEP 1
# -------------------------------------------------------

if st.session_state.step=="welcome":

    st.title("🚦 Traffic AI Monitoring System")
    st.subheader("מערכת בינה מלאכותית לניטור תנועה")

    video=st.file_uploader(
        "Upload Traffic Video / העלה סרטון תנועה",
        type=["mp4","mov","avi"]
    )

    if video:

        tfile=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        tfile.write(video.read())

        st.session_state.video_path=tfile.name
        st.session_state.step="config"
        st.rerun()

# -------------------------------------------------------
# STEP 2 CONFIG
# -------------------------------------------------------

elif st.session_state.step=="config":

    st.header("Analysis Settings / הגדרות ניתוח")

    location=st.text_input(
        "Location / מיקום",
        "Main Intersection / צומת ראשית"
    )

    duration=st.slider(
        "Analysis Duration Seconds / משך ניתוח",
        5,60,10
    )

    if st.button("🚀 START ANALYSIS / התחל ניתוח"):

        st.session_state.location=location
        st.session_state.duration=duration
        st.session_state.step="processing"
        st.rerun()

# -------------------------------------------------------
# STEP 3 PROCESSING
# -------------------------------------------------------

elif st.session_state.step=="processing":

    st.header("🧠 AI Processing / עיבוד בינה מלאכותית")

    col1,col2,col3=st.columns(3)

    total_metric=col1.empty()
    inbound_metric=col2.empty()
    outbound_metric=col3.empty()

    st.divider()

    model=YOLO("yolo11n.pt")

    cap=cv2.VideoCapture(st.session_state.video_path)

    fps=cap.get(cv2.CAP_PROP_FPS) or 30

    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames=int(st.session_state.duration*fps)

    raw=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4").name

    out=cv2.VideoWriter(
        raw,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w,h)
    )

    counted_ids=set()
    entry_pos={}
    prev_pos={}

    final_counts={"car":0,"bus":0,"truck":0,"motorcycle":0}

    direction_counts={"Inbound":0,"Outbound":0}

    speed_hist={"Inbound":[],"Outbound":[]}

    PXM=0.06

    results=model.track(
        source=st.session_state.video_path,
        stream=True,
        persist=True,
        imgsz=320
    )

    for i,r in enumerate(results):

        if i>=max_frames:
            break

        frame=r.orig_img.copy()

        if r.boxes.id is not None:

            boxes=r.boxes.xyxy.cpu().numpy()
            ids=r.boxes.id.int().cpu().tolist()
            clss=r.boxes.cls.int().cpu().tolist()
            confs=r.boxes.conf.cpu().numpy()

            for box,obj_id,cls,conf in zip(boxes,ids,clss,confs):

                label=model.names[cls]

                if label not in final_counts:
                    continue

                x1,y1,x2,y2=map(int,box)
                y_center=(y1+y2)/2

                # ---------- SPEED ----------
                if obj_id in prev_pos:

                    dy=abs(y_center-prev_pos[obj_id])
                    speed=(dy*PXM)/(1/fps)*3.6

                    if speed>2:

                        if y_center>h/2:
                            speed_hist["Inbound"].append(speed)
                        else:
                            speed_hist["Outbound"].append(speed)

                prev_pos[obj_id]=y_center

                # ---------- COUNT ----------
                if obj_id not in counted_ids:

                    counted_ids.add(obj_id)
                    final_counts[label]+=1
                    entry_pos[obj_id]=y_center

                else:

                    if obj_id in entry_pos:

                        start=entry_pos[obj_id]

                        if y_center>start+20:
                            direction_counts["Inbound"]+=1
                            entry_pos.pop(obj_id)

                        elif y_center<start-20:
                            direction_counts["Outbound"]+=1
                            entry_pos.pop(obj_id)

                # ---------- DRAW ----------
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                text=f"{label} {int(conf*100)}%"

                cv2.putText(
                    frame,
                    text,
                    (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        # ---------- TRAFFIC STATE ----------

        avg_in=np.mean(speed_hist["Inbound"]) if speed_hist["Inbound"] else 0
        avg_out=np.mean(speed_hist["Outbound"]) if speed_hist["Outbound"] else 0

        state_in,color_in=get_traffic_state(avg_in)
        state_out,color_out=get_traffic_state(avg_out)

        # ---------- HUD ----------

        sidebar=int(w*0.32)

        overlay=frame.copy()

        cv2.rectangle(
            overlay,
            (w-sidebar,0),
            (w,h),
            (0,0,0),
            -1
        )

        cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)

        cv2.putText(
            frame,
            st.session_state.location,
            (w-sidebar+20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        y=100

        for v,c in final_counts.items():

            cv2.putText(
                frame,
                f"{v.upper()}: {c}",
                (w-sidebar+20,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200,200,200),
                2
            )

            y+=35

        y+=20

        cv2.putText(
            frame,
            "TRAFFIC STATUS",
            (w-sidebar+20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

        y+=40

        cv2.putText(
            frame,
            f"INBOUND: {state_in}",
            (w-sidebar+20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_in,
            2
        )

        y+=35

        cv2.putText(
            frame,
            f"OUTBOUND: {state_out}",
            (w-sidebar+20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_out,
            2
        )

        # ---------- DASHBOARD ----------

        total=sum(final_counts.values())

        total_metric.metric(
            "Vehicles / רכבים",
            total
        )

        inbound_metric.metric(
            "Inbound Traffic / תנועה נכנסת",
            state_in
        )

        outbound_metric.metric(
            "Outbound Traffic / תנועה יוצאת",
            state_out
        )

        out.write(cv2.resize(frame,(w,h)))

    out.release()
    cap.release()

    final=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4").name

    clip=VideoFileClip(raw)

    clip.write_videofile(final,codec="libx264",audio=False)

    st.subheader("Processed Video / סרטון מעובד")

    st.video(final)

    with open(final,"rb") as f:

        st.download_button(
            "Download Video / הורד סרטון",
            f.read(),
            "traffic_report.mp4"
        )

    st.button(
        "New Analysis / ניתוח חדש",
        on_click=reset_app
    )
