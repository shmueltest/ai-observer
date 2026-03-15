import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Traffic Tracker AI", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0B0F14;
    color: #E5E7EB;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background-color: #111827;
    color: #E5E7EB;
    border: 1px solid #1F2937;
}

.stButton>button:hover {
    border: 1px solid #3B82F6;
    color: #3B82F6;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- RESET ----------------
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------- TRAFFIC STATE ----------------
def get_traffic_state(avg_speed):

    if avg_speed == 0:
        return "NO DATA / אין נתונים", (120,120,120)

    if avg_speed < 15:
        return "HEAVY / עומס כבד", (68,68,239)

    if avg_speed < 40:
        return "MODERATE / עומס בינוני", (21,204,250)

    return "LIGHT / תנועה קלה", (34,197,94)

# ---------------- SESSION STATE ----------------
if 'step' not in st.session_state:
    st.session_state.step = "welcome"

if 'config' not in st.session_state:
    st.session_state.config = {}

# ======================================================
# STEP 1 — WELCOME
# ======================================================

if st.session_state.step == "welcome":

    st.title("🚦 Traffic AI Intelligence System")
    st.subheader("מערכת בינה מלאכותית לניתוח תנועה")

    st.markdown("""
AI powered vehicle detection and congestion monitoring.  
מערכת לזיהוי רכבים וניתוח עומסי תנועה באמצעות בינה מלאכותית
""")

    uploaded_file = st.file_uploader(
        "Upload Traffic Video / העלה סרטון תנועה",
        type=['mp4','mov','avi']
    )

    if uploaded_file:

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())

        st.session_state.video_path = tfile.name
        st.session_state.step = "briefing"
        st.rerun()

# ======================================================
# STEP 2 — CONFIGURATION
# ======================================================

elif st.session_state.step == "briefing":

    st.header("📋 Analysis Configuration / הגדרות ניתוח")

    with st.container(border=True):

        loc = st.text_input(
            "Location Label / מיקום הצילום",
            "Main Intersection / צומת ראשית"
        )

        duration = st.slider(
            "Analysis Duration (seconds) / משך הניתוח (שניות)",
            1,
            60,
            10
        )

        c1, c2 = st.columns(2)

        with c1:

            burn_hud = st.toggle(
                "Burn Telemetry Into Video / הטמעת נתונים בסרטון",
                True
            )

            overlays = st.toggle(
                "Show Detection Boxes / הצג תיבות זיהוי",
                True
            )

        with c2:

            gen_graph = st.toggle(
                "Generate Graph / צור גרף",
                True
            )

            use_sidebar = st.toggle(
                "Sidebar Telemetry / נתונים בצד",
                True
            )

    if st.button("🚀 START ANALYSIS / התחל ניתוח"):

        st.session_state.config = {
            "loc":loc,
            "duration":duration,
            "burn":burn_hud,
            "overlays":overlays,
            "graph":gen_graph,
            "sidebar":use_sidebar
        }

        st.session_state.step = "processing"
        st.rerun()

# ======================================================
# STEP 3 — PROCESSING
# ======================================================

elif st.session_state.step == "processing":

    st.header("🧠 AI Processing Engine / מנוע ניתוח")

    # DASHBOARD METRICS
    m1,m2,m3,m4 = st.columns(4)

    total_metric = m1.empty()
    inbound_metric = m2.empty()
    outbound_metric = m3.empty()
    speed_metric = m4.empty()

    st.divider()

    try:

        model = YOLO("yolo11n.pt")

        cap = cv2.VideoCapture(st.session_state.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_frames = int(st.session_state.config['duration'] * fps)

        raw_path = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name

        out = cv2.VideoWriter(
            raw_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w,h)
        )

        tracked_ids=set()
        entry_points={}
        prev_pos={}

        final_counts={
            "car":0,
            "bus":0,
            "truck":0,
            "motorcycle":0
        }

        direction_counts={
            "Inbound":0,
            "Outbound":0
        }

        speed_history={
            "Inbound":[],
            "Outbound":[]
        }

        PXM=0.06

        with st.status("Running AI Detection / ניתוח בינה מלאכותית...", expanded=True):

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

                current_speeds={"Inbound":[],"Outbound":[]}

                if r.boxes.id is not None:

                    boxes=r.boxes.xyxy.cpu().numpy()
                    ids=r.boxes.id.int().cpu().tolist()
                    clss=r.boxes.cls.int().cpu().tolist()
                    conf=r.boxes.conf.cpu().numpy()

                    for box,obj_id,cls,cf in zip(boxes,ids,clss,conf):

                        label=model.names[cls]

                        if label in final_counts:

                            y_center=(box[1]+box[3])/2

                            if obj_id in prev_pos:

                                dy=abs(y_center-prev_pos[obj_id])
                                speed=(dy*PXM)/(1/fps)*3.6

                                if speed>2:

                                    if y_center>h/2:
                                        current_speeds["Inbound"].append(speed)
                                    else:
                                        current_speeds["Outbound"].append(speed)

                            prev_pos[obj_id]=y_center

                            if obj_id not in entry_points:

                                entry_points[obj_id]=y_center
                                final_counts[label]+=1
                                tracked_ids.add(obj_id)

                            if st.session_state.config['overlays']:

                                x1,y1,x2,y2=map(int,box)

                                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                for d in ["Inbound","Outbound"]:

                    if current_speeds[d]:
                        speed_history[d].append(np.mean(current_speeds[d]))

                # LIVE DASHBOARD UPDATE
                total=sum(final_counts.values())

                avg_in=np.mean(speed_history["Inbound"]) if speed_history["Inbound"] else 0
                avg_out=np.mean(speed_history["Outbound"]) if speed_history["Outbound"] else 0

                avg_speed=(avg_in+avg_out)/2 if (avg_in or avg_out) else 0

                total_metric.metric("🚗 Vehicles / רכבים", total)
                inbound_metric.metric("⬇ Inbound / כניסה", direction_counts["Inbound"])
                outbound_metric.metric("⬆ Outbound / יציאה", direction_counts["Outbound"])
                speed_metric.metric("⚡ Avg Speed / מהירות ממוצעת", f"{avg_speed:.1f} km/h")

                out.write(cv2.resize(frame,(w,h)))

        out.release()
        cap.release()

        final_path=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name

        clip=VideoFileClip(raw_path)

        clip.write_videofile(final_path,codec="libx264",audio=False)

        # RESULTS DASHBOARD
        st.divider()

        video_col,stats_col=st.columns([3,1])

        with video_col:

            st.subheader("📹 Processed Video / סרטון מעובד")

            st.video(final_path)

            with open(final_path,'rb') as f:

                st.download_button(
                    "📥 Download Report / הורד דו״ח",
                    f.read(),
                    "traffic_report.mp4",
                    "video/mp4"
                )

        with stats_col:

            st.subheader("📊 Vehicle Counts / ספירת רכבים")

            for v,c in final_counts.items():
                st.metric(v.capitalize(),c)

            st.metric("Inbound / כניסה",direction_counts["Inbound"])
            st.metric("Outbound / יציאה",direction_counts["Outbound"])

            st.button(
                "🔄 New Analysis / ניתוח חדש",
                on_click=reset_app
            )

        if st.session_state.config['graph']:

            st.subheader("📊 Vehicle Distribution / התפלגות רכבים")

            df=pd.DataFrame(
                list(final_counts.items()),
                columns=['Vehicle','Total']
            )

            st.bar_chart(df.set_index('Vehicle'))

    except Exception as e:

        st.error(f"System Error / שגיאת מערכת: {e}")

        if st.button("Emergency Reset / איפוס חירום"):
            reset_app()
