"""
traffic_tracker.py
==================
A Streamlit app that uses YOLOv11 object detection + ByteTrack multi-object
tracking to analyse traffic footage.

HOW IT WORKS — BIG PICTURE
───────────────────────────
1. The user uploads a video clip (MP4 / MOV / AVI).
2. Each frame is passed through a YOLO model that draws bounding boxes around
   vehicles and assigns each box a *persistent tracking ID* that stays the
   same across frames as long as the vehicle is visible.
3. We use those IDs to:
      a) COUNT unique vehicles  — a vehicle is counted the first time its ID
         appears, never again.
      b) DETERMINE DIRECTION    — we record where the vehicle's centre was
         when we first saw it, then watch which way it moves.  Once it has
         moved far enough (DIRECTION_THRESHOLD px), we lock its direction.
      c) ESTIMATE SPEED         — by measuring how many pixels the vehicle
         moves between consecutive frames, multiplying by the known real-world
         size of a pixel (PXM), and converting to km/h.
4. A HUD is optionally burned onto each frame showing live stats, and the
   processed video is re-encoded with libx264 for browser playback.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import atexit
import io
import os
import tempfile

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from moviepy import VideoFileClip
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# Centralise all "magic numbers" here so they are easy to tune.
# ═══════════════════════════════════════════════════════════════════════════════

# Metres-per-pixel: how many real-world metres one pixel represents vertically.
# This depends entirely on the camera height / angle / zoom.
# 0.06 m/px is a reasonable default for a camera mounted ~6–8 m above a road.
# Increase this value if speeds appear too low; decrease if too high.
PXM: float = 0.06

# How many pixels a vehicle must travel vertically from its entry position
# before we commit to calling it "Inbound" or "Outbound".
# Too small → jitter causes wrong counts.  Too large → short clips miss vehicles.
DIRECTION_THRESHOLD: int = 20

# Speeds below this value (km/h) are treated as tracking noise and ignored.
MIN_SPEED_KMH: float = 2.0

# The HUD panel on the right edge of each frame is this fraction of frame width.
HUD_WIDTH_RATIO: float = 0.30

# Only these COCO class names are counted; everything else is ignored.
VEHICLE_CLASSES: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")

# Minimum YOLO confidence required to act on a detection.
# Lower = more detections but more false positives.
MIN_CONFIDENCE: float = 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# BILINGUAL UI STRINGS
# Keep all user-visible text here so the rest of the code stays clean.
# Format: "English text | טקסט עברי"
# ═══════════════════════════════════════════════════════════════════════════════
T = {
    # App-level
    "app_title":        "🚦 Traffic Tracking System | מערכת ניטור תנועה",
    "authors":          "שמואל קויפמאן וישי גפני | Shmuel Koyfman & Yishai Gafni",
    "yt_hint":          "📥 Download YouTube videos here | הורדת סרטוני יוטיוב כאן",

    # Welcome step
    "upload_label":     "Upload traffic footage | העלה סרטון תנועה",
    "upload_help":      "Supported formats: MP4, MOV, AVI | פורמטים נתמכים: MP4, MOV, AVI",

    # Briefing step
    "briefing_header":  "📋 Analysis Setup | הגדרת ניתוח",
    "loc_label":        "Location label | תווית מיקום",
    "loc_default":      "Sector 7G – Main Intersection",
    "duration_label":   "Analysis window (seconds) | חלון ניתוח (שניות)",
    "toggle_burn":      "Burn telemetry into video | שרוף נתונים על הווידאו",
    "toggle_overlays":  "Show tracking boxes | הצג תיבות מעקב",
    "toggle_graph":     "Show results chart | הצג גרף תוצאות",
    "toggle_sidebar":   "Live sidebar metrics | מדדים חיים בסרגל צד",
    "start_btn":        "🚀 Start Analysis | התחל ניתוח",

    # Processing step
    "proc_header":      "🧠 Analysing footage… | מנתח סרטון…",
    "proc_status":      "Running YOLO + tracking… | מריץ YOLO + מעקב…",
    "proc_done":        "✅ Analysis complete | ניתוח הושלם",
    "proc_encode":      "Re-encoding video for browser… | מקודד מחדש לדפדפן…",

    # Results
    "results_header":   "📊 Results | תוצאות",
    "dl_video":         "📥 Download video | הורד ווידאו",
    "dl_csv":           "📄 Download CSV | הורד CSV",
    "chart_title":      "Vehicle distribution | התפלגות כלי רכב",
    "new_analysis":     "🔄 New analysis | ניתוח חדש",

    # Sidebar
    "sidebar_title":    "📡 Telemetry | טלמטריה",
    "unique_vehicles":  "Unique vehicles | רכבים ייחודיים",
    "inbound":          "Inbound | נכנסים",
    "outbound":         "Outbound | יוצאים",
    "traffic_state":    "Traffic state | מצב תנועה",
    "speed_label":      "Avg speed | מהירות ממוצעת",

    # Errors
    "err_video":        "❌ Cannot open video file. Please re-upload. | לא ניתן לפתוח קובץ. נסה שוב.",
    "err_generic":      "System error | שגיאת מערכת",
    "err_reset":        "🔄 Emergency reset | איפוס חירום",
}


# ═══════════════════════════════════════════════════════════════════════════════
# APP CONFIG & GLOBAL STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Traffic Tracker | ניטור תנועה",
    page_icon="🚦",
    layout="wide",
)

# Custom CSS — improves readability over the default Streamlit theme.
# Dark navy background, white/light text, and red accent colours.
st.markdown("""
<style>
/* ── Page background & base text ── */
.stApp {
    background: linear-gradient(160deg, #1a2540 0%, #243052 100%);
    color: #e8eaf0;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #fff; }

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background: #1e2d50;
    color: #e8eaf0;
    border: 1px solid #3a4a6b;
    font-size: 0.95rem;
    transition: border-color 0.2s, color 0.2s;
}
.stButton > button:hover { border-color: #ff4b4b; color: #ff4b4b; }

/* ── Bordered containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    padding: 1rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.25);
    border: 1px solid #db2d2d;
    border-radius: 8px;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMP FILE MANAGEMENT
# Every NamedTemporaryFile we create is registered here.
# Python's atexit hook deletes them all when the process exits so we don't
# slowly fill the server's disk.
# ═══════════════════════════════════════════════════════════════════════════════
_tmp_files: list[str] = []


def _register_tmp(path: str) -> str:
    """Register a temp file path for cleanup on exit, then return it."""
    _tmp_files.append(path)
    return path


atexit.register(lambda: [os.unlink(f) for f in _tmp_files if os.path.exists(f)])


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def reset_app() -> None:
    """Wipe all session state and restart from the welcome screen."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def get_traffic_state(avg_speed: float) -> tuple[str, tuple[int, int, int]]:
    """
    Classify traffic density from average speed (km/h).

    Returns a (label, BGR_colour) tuple used in the HUD and sidebar.
    Speed thresholds are typical urban road values:
      < 15 km/h  → heavy congestion
      15–40 km/h → moderate flow
      > 40 km/h  → light / free flow
    """
    if avg_speed == 0:
        return "NO DATA",  (200, 200, 200)   # grey  — no vehicles seen yet
    if avg_speed < 15:
        return "HEAVY",    (0,   0, 255)     # red   — congestion
    if avg_speed < 40:
        return "MODERATE", (0, 255, 255)     # yellow — moderate
    return     "LIGHT",    (0, 255,   0)     # green  — free flow


@st.cache_resource(show_spinner="Loading YOLO model… | טוען מודל…")
def load_model() -> YOLO:
    """
    Load the YOLOv11-nano model and cache it for the whole session.

    @st.cache_resource means this function runs ONCE no matter how many times
    Streamlit re-runs the script.  Without caching, every page interaction
    would reload ~6 MB of model weights from disk.

    'yolo11n.pt' is the nano variant — fastest inference, lowest accuracy.
    Swap for 'yolo11s.pt' or 'yolo11m.pt' for better accuracy at the cost
    of speed.
    """
    return YOLO("yolo11n.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# Streamlit re-runs this entire script on every user interaction.
# session_state is the only place data persists across re-runs.
# ═══════════════════════════════════════════════════════════════════════════════
if "step" not in st.session_state:
    st.session_state.step = "welcome"   # flow: "welcome" → "briefing" → "processing"
if "config" not in st.session_state:
    st.session_state.config = {}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == "welcome":

    st.title(T["app_title"])
    st.caption(T["authors"])
    st.divider()

    col_info, col_upload = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown("### How it works | איך זה עובד")
        st.markdown("""
        1. **Upload** a traffic video clip  
           **העלה** קליפ וידאו של תנועה

        2. **Configure** the analysis settings  
           **הגדר** את הגדרות הניתוח

        3. **Download** the annotated video + CSV report  
           **הורד** את הווידאו המוערך + דוח CSV
        """)
        st.info(f"[{T['yt_hint']}](https://en1.savefrom.net/1-youtube-video-downloader-13sg/)")

    with col_upload:
        st.markdown("### Upload footage | העלאת סרטון")
        uploaded_file = st.file_uploader(
            T["upload_label"],
            type=["mp4", "mov", "avi"],
            help=T["upload_help"],
        )
        if uploaded_file:
            # Save to a temp file so OpenCV and YOLO can read it by path.
            # Streamlit's UploadedFile is an in-memory buffer, not a real path.
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            _register_tmp(tfile.name)
            st.session_state.video_path = tfile.name
            st.session_state.step = "briefing"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANALYSIS BRIEFING / SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "briefing":

    st.title(T["briefing_header"])
    st.divider()

    # Preview the uploaded video so the user can confirm it loaded correctly
    with st.expander("📹 Preview uploaded footage | תצוגה מקדימה של הסרטון", expanded=True):
        st.video(st.session_state.video_path)

    st.markdown("### Settings | הגדרות")

    with st.container(border=True):
        loc = st.text_input(
            T["loc_label"],
            value=T["loc_default"],
            help="This label appears in the HUD overlay | תווית זו מופיעה על גבי הווידאו",
        )
        duration = st.slider(
            T["duration_label"],
            min_value=1, max_value=60, value=10,
            help="Longer = more data, slower processing | ארוך יותר = יותר נתונים, עיבוד איטי יותר",
        )

        st.markdown("#### Output options | אפשרויות פלט")
        c1, c2 = st.columns(2)
        with c1:
            burn_hud    = st.toggle(T["toggle_burn"],     value=True)
            overlays    = st.toggle(T["toggle_overlays"], value=True)
        with c2:
            gen_graph   = st.toggle(T["toggle_graph"],   value=True)
            use_sidebar = st.toggle(T["toggle_sidebar"], value=True)

    st.divider()
    if st.button(T["start_btn"], use_container_width=True):
        st.session_state.config = {
            "loc": loc, "duration": duration,
            "burn": burn_hud, "overlays": overlays,
            "graph": gen_graph, "sidebar": use_sidebar,
        }
        st.session_state.step = "processing"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PROCESSING
# This is the core of the app.  It runs once and stores results in session_state.
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "processing":

    st.title(T["proc_header"])

    try:
        # ── Load the YOLO model (cached — instant on second run) ──────────────
        model = load_model()

        # ── Read video metadata then release immediately ───────────────────────
        # We open the file ONLY to read fps/width/height, then close it.
        # model.track() will open its own internal reader later, and keeping
        # two handles open simultaneously can cause conflicts on Windows.
        cap = cv2.VideoCapture(st.session_state.video_path)
        if not cap.isOpened():
            st.error(T["err_video"])
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()   # ← release right away

        max_frames = int(st.session_state.config["duration"] * fps)

        # ── Set up the output video writer ────────────────────────────────────
        # We first write a raw mp4v file (fast, OpenCV-native), then re-encode
        # with libx264 at the end for browser compatibility.
        raw_path = _register_tmp(
            tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        )
        out = cv2.VideoWriter(
            raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        # ── Tracking data structures ──────────────────────────────────────────
        #
        # entry_points   {obj_id: y_center_at_first_sight}
        #   Stores where each vehicle first appeared vertically.
        #   Sentinel values mark locked direction:
        #     99999  → confirmed Inbound  (moving downward in frame)
        #    -99999  → confirmed Outbound (moving upward in frame)
        #
        # prev_pos       {obj_id: y_center_last_frame}
        #   Used to compute per-frame vertical displacement for speed estimation.
        #
        # counted_ids    set of obj_ids already counted
        #   An ID is added the FIRST time we see it with sufficient confidence.
        #   We NEVER count the same ID twice, even if the vehicle disappears and
        #   reappears (ByteTrack re-assigns the same ID on re-entry within a
        #   short gap window).
        #
        entry_points: dict[int, float] = {}
        prev_pos:     dict[int, float] = {}
        counted_ids:  set[int]         = set()

        final_counts     = dict.fromkeys(VEHICLE_CLASSES, 0)
        direction_counts = {"Inbound": 0, "Outbound": 0}
        speed_history    = {"Inbound": [], "Outbound": []}

        # ── Main processing loop ──────────────────────────────────────────────
        with st.status(T["proc_status"], expanded=True) as status:
            progress_bar = st.progress(0.0)
            frame_info   = st.empty()   # placeholder for live frame counter

            # model.track() is a generator: it yields one Results object per
            # frame without loading the entire video into memory at once.
            # persist=True  → ByteTrack keeps IDs stable across frames.
            # stream=True   → memory-efficient generator mode.
            # imgsz=320     → resize frames to 320 px before inference (faster).
            # conf=          → ignore detections below this confidence.
            results = model.track(
                source=st.session_state.video_path,
                stream=True,
                persist=True,
                imgsz=320,
                conf=MIN_CONFIDENCE,
            )

            for i, r in enumerate(results):
                if i >= max_frames:
                    break

                # Update progress UI every 10 frames to reduce Streamlit overhead
                if i % 10 == 0:
                    progress_bar.progress(min(i / max_frames, 1.0))
                    frame_info.caption(
                        f"Frame {i}/{max_frames} | פריים {i}/{max_frames}"
                    )

                frame = r.orig_img.copy()
                current_speeds: dict[str, list[float]] = {"Inbound": [], "Outbound": []}

                # r.boxes contains all detections for this frame.
                # r.boxes.id is None when no tracks exist in this frame.
                if r.boxes.id is not None:

                    # Pull all per-box data off GPU in one batch operation.
                    # Calling .cpu() inside a per-box loop is significantly slower.
                    boxes       = r.boxes.xyxy.cpu().numpy()        # [x1,y1,x2,y2]
                    ids         = r.boxes.id.int().cpu().tolist()   # tracking IDs
                    clss        = r.boxes.cls.int().cpu().tolist()  # class indices
                    confidences = r.boxes.conf.cpu().numpy()        # 0.0 – 1.0

                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confidences):

                        label = model.names[cls]

                        # Skip non-vehicle detections and low-confidence ones
                        if label not in VEHICLE_CLASSES or conf < MIN_CONFIDENCE:
                            continue

                        # Vertical centre of the bounding box.
                        # We use the Y axis because most traffic cameras are
                        # overhead/angled and vehicles primarily move up or down.
                        y_center = (box[1] + box[3]) / 2.0

                        # ── SPEED ESTIMATION ───────────────────────────────────
                        # Physics:
                        #   displacement (m) = pixel_delta × PXM
                        #   speed (m/s)      = displacement × fps
                        #   speed (km/h)     = speed_ms × 3.6
                        #
                        # We only measure vertical displacement (dy) because that
                        # is the dominant motion axis for overhead camera footage.
                        if obj_id in prev_pos:
                            dy        = abs(y_center - prev_pos[obj_id])
                            speed_ms  = dy * PXM * fps
                            speed_kmh = speed_ms * 3.6
                            if speed_kmh > MIN_SPEED_KMH:
                                bucket = "Inbound" if y_center > h / 2 else "Outbound"
                                current_speeds[bucket].append(speed_kmh)

                        prev_pos[obj_id] = y_center

                        # ── UNIQUE VEHICLE COUNT ───────────────────────────────
                        # A vehicle is counted EXACTLY ONCE — the first frame its
                        # tracking ID appears.  counted_ids prevents any re-count
                        # even if ByteTrack briefly loses and re-finds the vehicle.
                        if obj_id not in counted_ids:
                            counted_ids.add(obj_id)
                            entry_points[obj_id] = y_center
                            final_counts[label] += 1

                        # ── DIRECTION DETECTION ────────────────────────────────
                        # Compare current Y to the entry Y recorded above.
                        #   Moving DOWN (increasing Y) → Inbound
                        #   Moving UP   (decreasing Y) → Outbound
                        #
                        # The sentinel lock (99999 / -99999) ensures direction is
                        # counted ONCE per vehicle, no matter how many frames it
                        # takes to travel past the threshold.
                        if obj_id in entry_points:
                            ref = entry_points[obj_id]
                            if ref not in (99999, -99999):
                                if y_center > ref + DIRECTION_THRESHOLD:
                                    direction_counts["Inbound"] += 1
                                    entry_points[obj_id] = 99999    # locked ↓
                                elif y_center < ref - DIRECTION_THRESHOLD:
                                    direction_counts["Outbound"] += 1
                                    entry_points[obj_id] = -99999   # locked ↑

                        # ── BOUNDING BOX OVERLAY ───────────────────────────────
                        if st.session_state.config["overlays"]:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                            cv2.putText(
                                frame,
                                f"{label} #{obj_id} {int(conf * 100)}%",
                                (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1,
                            )

                # ── Aggregate per-frame speed samples into running history ─────
                for d in ("Inbound", "Outbound"):
                    if current_speeds[d]:
                        speed_history[d].append(float(np.mean(current_speeds[d])))

                # ── HUD BURN-IN ────────────────────────────────────────────────
                # Draw a semi-transparent dark panel on the right of the frame,
                # then paint text statistics on top of it.
                # cv2.addWeighted blends overlay (0.70) with original frame (0.30).
                if st.session_state.config["burn"]:
                    sidebar_w = int(w * HUD_WIDTH_RATIO)
                    x_hud     = w - sidebar_w
                    overlay   = frame.copy()
                    cv2.rectangle(overlay, (x_hud, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

                    cv2.putText(
                        frame, st.session_state.config["loc"].upper(),
                        (x_hud + 10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1,
                    )
                    y_off = 70
                    for obj, val in final_counts.items():
                        cv2.putText(
                            frame, f"{obj.upper()}: {val}",
                            (x_hud + 10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 255), 1,
                        )
                        y_off += 26

                    avg_in  = speed_history["Inbound"][-1]  if speed_history["Inbound"]  else 0.0
                    avg_out = speed_history["Outbound"][-1] if speed_history["Outbound"] else 0.0
                    state_in,  c_in  = get_traffic_state(avg_in)
                    state_out, c_out = get_traffic_state(avg_out)

                    y_off += 15
                    cv2.putText(
                        frame, "TRAFFIC STATUS",
                        (x_hud + 10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1,
                    )
                    y_off += 28
                    cv2.putText(
                        frame, f"IN:  {state_in}",
                        (x_hud + 10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, c_in, 2,
                    )
                    y_off += 26
                    cv2.putText(
                        frame, f"OUT: {state_out}",
                        (x_hud + 10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, c_out, 2,
                    )
                    cv2.putText(
                        frame, f"Frame {i+1}/{max_frames}",
                        (x_hud + 10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1,
                    )

                out.write(frame)   # write annotated frame to output video

            # ── Finalise output ────────────────────────────────────────────────
            out.release()
            progress_bar.progress(1.0)
            frame_info.empty()

            # Re-encode with libx264 so browsers can play the video.
            # mp4v (MPEG-4 Part 2) is not universally supported in browsers;
            # H.264 (libx264) is the universal standard.
            status.update(label=T["proc_encode"], state="running")
            final_path = _register_tmp(
                tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            )
            clip = VideoFileClip(raw_path)
            clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)
            clip.close()

            status.update(label=T["proc_done"], state="complete")

        # Persist results so the UI below survives future Streamlit re-runs
        st.session_state.speed_history    = speed_history
        st.session_state.final_counts     = final_counts
        st.session_state.direction_counts = direction_counts
        st.session_state.unique_count     = len(counted_ids)
        st.session_state.final_path       = final_path

        # ══════════════════════════════════════════════════════════════════════
        # RESULTS DISPLAY
        # ══════════════════════════════════════════════════════════════════════

        st.title(T["results_header"])
        st.divider()

        # ── Top KPI metrics row ────────────────────────────────────────────────
        avg_spd_in  = round(np.mean(speed_history["Inbound"]),  1) if speed_history["Inbound"]  else 0.0
        avg_spd_out = round(np.mean(speed_history["Outbound"]), 1) if speed_history["Outbound"] else 0.0
        total       = sum(final_counts.values())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total vehicles | סה״כ רכבים",        total)
        k2.metric("Unique IDs | מזהים ייחודיים",         len(counted_ids))
        k3.metric("Inbound | נכנסים",                    direction_counts["Inbound"])
        k4.metric("Outbound | יוצאים",                   direction_counts["Outbound"])
        k5.metric("Avg speed (km/h) | מהירות ממוצעת",   f"{max(avg_spd_in, avg_spd_out):.1f}")

        st.divider()

        # ── Video + downloads + chart ──────────────────────────────────────────
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("#### Annotated video | ווידאו עם הערות")
            st.video(final_path)

            dl1, dl2 = st.columns(2)
            with dl1:
                with open(final_path, "rb") as f:
                    st.download_button(
                        T["dl_video"], f.read(),
                        "traffic_report.mp4", "video/mp4",
                        use_container_width=True,
                    )
            with dl2:
                df_report = pd.DataFrame({
                    "Category | קטגוריה": (
                        list(final_counts.keys()) + ["Inbound", "Outbound"]
                    ),
                    "Count | ספירה": (
                        list(final_counts.values()) +
                        [direction_counts["Inbound"], direction_counts["Outbound"]]
                    ),
                })
                buf = io.StringIO()
                df_report.to_csv(buf, index=False)
                st.download_button(
                    T["dl_csv"], buf.getvalue(),
                    "traffic_report.csv", "text/csv",
                    use_container_width=True,
                )

        with col_right:
            if st.session_state.config["graph"]:
                st.markdown(f"#### {T['chart_title']}")
                df_chart = pd.DataFrame(
                    list(final_counts.items()),
                    columns=["Vehicle | רכב", "Count | ספירה"],
                )
                st.bar_chart(df_chart.set_index("Vehicle | רכב"))

            st.markdown("#### Breakdown | פירוט")
            for vtype, count in final_counts.items():
                ca, cb = st.columns([2, 1])
                ca.write(vtype.capitalize())
                cb.write(f"**{count}**")

            st.divider()
            st.button(T["new_analysis"], on_click=reset_app, use_container_width=True)

        # ── Sidebar telemetry ──────────────────────────────────────────────────
        if st.session_state.config["sidebar"]:
            avg_in  = speed_history["Inbound"][-1]  if speed_history["Inbound"]  else 0.0
            avg_out = speed_history["Outbound"][-1] if speed_history["Outbound"] else 0.0

            st.sidebar.title(T["sidebar_title"])
            st.sidebar.metric(T["unique_vehicles"], len(counted_ids))
            st.sidebar.divider()

            st.sidebar.subheader(f"⬇️ {T['inbound']}")
            st.sidebar.write(f"**{T['traffic_state']}:** {get_traffic_state(avg_in)[0]}")
            st.sidebar.metric(T["speed_label"], f"{avg_in:.1f} km/h")
            st.sidebar.metric("Count | ספירה", direction_counts["Inbound"])

            st.sidebar.divider()
            st.sidebar.subheader(f"⬆️ {T['outbound']}")
            st.sidebar.write(f"**{T['traffic_state']}:** {get_traffic_state(avg_out)[0]}")
            st.sidebar.metric(T["speed_label"], f"{avg_out:.1f} km/h")
            st.sidebar.metric("Count | ספירה", direction_counts["Outbound"])

    except Exception as e:
        st.error(f"{T['err_generic']}: {e}")
        if st.button(T["err_reset"]):
            reset_app()
