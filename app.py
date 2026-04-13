"""
traffic_tracker.py
==================
This app watches a traffic video and counts the cars.

Here is what it does, step by step:
1. The user uploads a video.
2. The app looks at every frame and finds all the vehicles in it.
   Each vehicle gets a number (an ID) that sticks to it across frames.
3. Using those IDs the app:
   - Counts each vehicle once (the first time it appears).
   - Figures out which way it is going (into the scene or out of it).
   - Estimates how fast it is moving.
4. The results are drawn on top of the video and shown to the user.
"""

# --- Built-in Python tools ---
import atexit    # lets us run cleanup code when the app closes
import io        # for building files in memory (used for CSV export)
import os        # for deleting temp files
import tempfile  # for creating temp files on disk

# --- Installed packages ---
import cv2                    # reads and writes video frames
import numpy as np            # maths on arrays of numbers
import pandas as pd           # builds tables (used for the CSV export)
import streamlit as st        # builds the web UI
from moviepy import VideoFileClip   # re-encodes the output video
from ultralytics import YOLO        # the AI model that finds vehicles


# =============================================================================
# SETTINGS
# Change these numbers to tune how the app behaves.
# =============================================================================

# How many real-world metres one pixel represents (top-to-bottom).
# If the speed numbers look wrong, adjust this first.
# ~0.06 works for a camera about 6-8 metres above the road.
PXM: float = 0.06

# How many pixels a vehicle must move before we decide which way it is going.
# If direction counts seem wrong, try raising this a little.
DIRECTION_THRESHOLD: int = 20

# We ignore speed readings below this (km/h) — they are just camera wobble.
MIN_SPEED_KMH: float = 2.0

# The info panel drawn on the right side of the video takes up this much
# of the frame width. 0.30 means 30%.
HUD_WIDTH_RATIO: float = 0.30

# Only these vehicle types are counted. Everything else is ignored.
VEHICLE_CLASSES: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")

# The AI model must be this confident before we trust a detection.
# Lower = catches more vehicles but also makes more mistakes.
MIN_CONFIDENCE: float = 0.35


# =============================================================================
# ALL UI TEXT (English and Hebrew side by side)
# Keeping all the words here means we only need to change one place
# if we want to update a label.
# =============================================================================
T = {
    # App title and authors
    "app_title":        "🚦 Traffic Tracking System | מערכת ניטור תנועה",
    "authors":          "שמואל קויפמאן וישי גפני | Shmuel Koufman & Yishai Gafni",
    "yt_hint":          "📥 Download YouTube videos here | הורדת סרטוני יוטיוב כאן",

    # Step 1 — upload screen
    "upload_label":     "Upload traffic footage | העלה סרטון תנועה",
    "upload_help":      "Supported formats: MP4, MOV, AVI | פורמטים נתמכים: MP4, MOV, AVI",

    # Step 2 — settings screen
    "briefing_header":  "📋 Analysis Setup | הגדרת ניתוח",
    "loc_label":        "Location label | תווית מיקום",
    "loc_default":      "Sector 7G – Main Intersection",
    "duration_label":   "How many seconds to analyse | כמה שניות לנתח",
    "toggle_burn":      "Draw stats onto the video | צייר סטטיסטיקות על הווידאו",
    "toggle_overlays":  "Show boxes around vehicles | הצג תיבות סביב רכבים",
    "toggle_graph":     "Show results chart | הצג גרף תוצאות",
    "toggle_sidebar":   "Show live numbers in the sidebar | הצג מספרים חיים בסרגל צד",
    "start_btn":        "🚀 Start Analysis | התחל ניתוח",

    # Step 3 — processing screen
    "proc_header":      "🧠 Analysing footage… | מנתח סרטון…",
    "proc_status":      "Finding and counting vehicles… | מוצא וסופר רכבים…",
    "proc_done":        "✅ Done! | סיימנו!",
    "proc_encode":      "Preparing video for playback… | מכין ווידאו להפעלה…",

    # Results screen
    "results_header":   "📊 Results | תוצאות",
    "dl_video":         "📥 Download video | הורד ווידאו",
    "dl_csv":           "📄 Download CSV | הורד CSV",
    "chart_title":      "Vehicles by type | רכבים לפי סוג",
    "new_analysis":     "🔄 Analyse another video | נתח סרטון אחר",

    # Sidebar
    "sidebar_title":    "📡 Live Numbers | מספרים בזמן אמת",
    "unique_vehicles":  "Vehicles seen | רכבים שנראו",
    "inbound":          "Coming in | נכנסים",
    "outbound":         "Going out | יוצאים",
    "traffic_state":    "Traffic | תנועה",
    "speed_label":      "Avg speed | מהירות ממוצעת",

    # Error messages
    "err_video":        "❌ Could not open the video. Please try uploading it again. | לא ניתן לפתוח את הסרטון. נסה שוב.",
    "err_generic":      "Something went wrong | משהו השתבש",
    "err_reset":        "🔄 Start over | התחל מחדש",
}


# =============================================================================
# PAGE SETUP AND VISUAL STYLE
# =============================================================================

st.set_page_config(
    page_title="Traffic Tracker | ניטור תנועה",
    page_icon="🚦",
    layout="wide",
)

# The CSS below changes colours, card styles, and button looks.
# It runs once when the page loads and affects the whole app.
st.markdown("""
<style>
/* Dark blue background for the whole page */
.stApp {
    background: linear-gradient(160deg, #1a2540 0%, #243052 100%);
    color: #e8eaf0;
}

/* Number cards (the big stat boxes) */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #fff; }

/* Buttons */
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

/* Boxes with a border (st.container with border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    padding: 1rem !important;
}

/* Collapsible sections */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.25);
    border: 1px solid #db2d2d;
    border-radius: 8px;
}

/* Horizontal lines */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Left sidebar panel */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# TEMP FILE CLEANUP
# Every time we create a temporary file we add it to this list.
# When the app closes, Python automatically deletes them all.
# =============================================================================
_tmp_files: list[str] = []


def _register_tmp(path: str) -> str:
    """Add a file to the cleanup list, then return its path."""
    _tmp_files.append(path)
    return path


# This runs the cleanup when the app process ends
atexit.register(lambda: [os.unlink(f) for f in _tmp_files if os.path.exists(f)])


# =============================================================================
# SMALL HELPER FUNCTIONS
# =============================================================================

def reset_app() -> None:
    """Clear everything and go back to the start screen."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def get_traffic_state(avg_speed: float) -> tuple[str, tuple[int, int, int]]:
    """
    Turn a speed (km/h) into a traffic label and a colour.

    Returns two things: a word like "HEAVY" and a colour for the video overlay.
      No data yet  → grey
      Under 15     → HEAVY (red)
      15 to 40     → MODERATE (yellow)
      Over 40      → LIGHT (green)
    """
    if avg_speed == 0:
        return "NO DATA",  (200, 200, 200)  # grey
    if avg_speed < 15:
        return "HEAVY",    (0,   0, 255)    # red
    if avg_speed < 40:
        return "MODERATE", (0, 255, 255)    # yellow
    return     "LIGHT",    (0, 255,   0)    # green


@st.cache_resource(show_spinner="Loading AI model… | טוען מודל בינה מלאכותית…")
def load_model() -> YOLO:
    """
    Load the vehicle-detection model.

    The @st.cache_resource tag means this only runs ONCE per session.
    Without it the model would reload from disk every time the user
    clicks anything, which would be very slow.

    'yolo11n.pt' is the small fast version of the model.
    Use 'yolo11s.pt' or 'yolo11m.pt' if you want better accuracy
    and do not mind waiting a bit longer.
    """
    return YOLO("yolo11n.pt")


# =============================================================================
# REMEMBER WHERE WE ARE
# Streamlit re-runs this whole file every time the user clicks something.
# We use st.session_state to remember things between those re-runs.
# 'step' tracks which screen to show: welcome → settings → processing.
# =============================================================================
if "step" not in st.session_state:
    st.session_state.step = "welcome"
if "config" not in st.session_state:
    st.session_state.config = {}


# =============================================================================
# SCREEN 1 — WELCOME
# =============================================================================
if st.session_state.step == "welcome":

    st.title(T["app_title"])
    st.caption(T["authors"])
    st.divider()

    # Two columns: instructions on the left, file upload on the right
    col_info, col_upload = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown("### How it works | איך זה עובד")
        st.markdown("""
        1. **Upload** a traffic video clip  
           **העלה** קליפ וידאו של תנועה

        2. **Choose** your settings  
           **בחר** את ההגדרות שלך

        3. **Download** the result video and data  
           **הורד** את הווידאו והנתונים
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
            # Streamlit gives us the file as raw bytes in memory.
            # We save it to a real file on disk so the AI model can open it.
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            _register_tmp(tfile.name)
            st.session_state.video_path = tfile.name
            st.session_state.step = "briefing"
            st.rerun()


# =============================================================================
# SCREEN 2 — SETTINGS
# =============================================================================
elif st.session_state.step == "briefing":

    st.title(T["briefing_header"])
    st.divider()

    # Let the user check the video loaded correctly before starting
    with st.expander("📹 Preview uploaded footage | תצוגה מקדימה של הסרטון", expanded=True):
        st.video(st.session_state.video_path)

    st.markdown("### Settings | הגדרות")

    with st.container(border=True):
        loc = st.text_input(
            T["loc_label"],
            value=T["loc_default"],
            help="This text will appear in the corner of the output video. | הטקסט הזה יופיע בפינת הווידאו.",
        )
        duration = st.slider(
            T["duration_label"],
            min_value=1, max_value=60, value=10,
            help="More seconds = more data, but takes longer. | יותר שניות = יותר נתונים, אבל לוקח יותר זמן.",
        )

        st.markdown("#### Output options | אפשרויות פלט")
        c1, c2 = st.columns(2)
        with c1:
            burn_hud  = st.toggle(T["toggle_burn"],     value=True)
            overlays  = st.toggle(T["toggle_overlays"], value=True)
        with c2:
            gen_graph   = st.toggle(T["toggle_graph"],   value=True)
            use_sidebar = st.toggle(T["toggle_sidebar"], value=True)

    st.divider()
    if st.button(T["start_btn"], use_container_width=True):
        # Save the settings so the next screen can read them
        st.session_state.config = {
            "loc": loc, "duration": duration,
            "burn": burn_hud, "overlays": overlays,
            "graph": gen_graph, "sidebar": use_sidebar,
        }
        st.session_state.step = "processing"
        st.rerun()


# =============================================================================
# SCREEN 3 — PROCESSING
# This is the main engine. It goes through the video frame by frame,
# finds all the vehicles, and builds up the counts and speed data.
# =============================================================================
elif st.session_state.step == "processing":

    st.title(T["proc_header"])

    try:
        # Load the AI model (uses the cached version if already loaded)
        model = load_model()

        # Open the video just to read its width, height, and frame rate,
        # then close it straight away. The AI model will open its own
        # copy of the video later — two open handles at once can cause
        # problems on some computers.
        cap = cv2.VideoCapture(st.session_state.video_path)
        if not cap.isOpened():
            st.error(T["err_video"])
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # frames per second
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()  # done — close it now

        # How many frames to process in total
        max_frames = int(st.session_state.config["duration"] * fps)

        # Set up a file to write the output video into.
        # We use a raw format first (mp4v) because it is fast to write.
        # Later we convert it to the format browsers can play (h264).
        raw_path = _register_tmp(
            tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        )
        out = cv2.VideoWriter(
            raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        # --- Data we collect while watching the video ---
        #
        # entry_points: where each vehicle was when we first saw it (y position).
        #   We use two special values to mark that we have already decided
        #   its direction:  99999 = going inbound,  -99999 = going outbound.
        #
        # prev_pos: where each vehicle was in the previous frame.
        #   We use this to measure how far it moved (for speed).
        #
        # counted_ids: the set of vehicle IDs we have already counted.
        #   Once an ID is in here we never count it again, even if it
        #   disappears and comes back.
        entry_points: dict[int, float] = {}
        prev_pos:     dict[int, float] = {}
        counted_ids:  set[int]         = set()

        final_counts     = dict.fromkeys(VEHICLE_CLASSES, 0)  # e.g. {"car": 0, ...}
        direction_counts = {"Inbound": 0, "Outbound": 0}
        speed_history    = {"Inbound": [], "Outbound": []}     # one entry per frame

        # --- Main loop — go through every frame ---
        with st.status(T["proc_status"], expanded=True) as status:
            progress_bar = st.progress(0.0)
            frame_info   = st.empty()  # small text showing current frame number

            # model.track() gives us one frame at a time.
            # persist=True  → keeps the same ID on the same vehicle across frames.
            # stream=True   → does not load the whole video into memory at once.
            # imgsz=320     → shrinks each frame to 320px before the AI looks at it (faster).
            # conf=          → skip detections the model is not sure about.
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

                # Update the progress bar every 10 frames to keep things snappy
                if i % 10 == 0:
                    progress_bar.progress(min(i / max_frames, 1.0))
                    frame_info.caption(f"Frame {i}/{max_frames} | פריים {i}/{max_frames}")

                frame = r.orig_img.copy()  # grab the pixel data for this frame
                current_speeds: dict[str, list[float]] = {"Inbound": [], "Outbound": []}

                # r.boxes holds all the vehicles found in this frame.
                # r.boxes.id is empty when nothing was found.
                if r.boxes.id is not None:

                    # Read all the box data in one go (faster than one by one)
                    boxes       = r.boxes.xyxy.cpu().numpy()        # corners: x1,y1,x2,y2
                    ids         = r.boxes.id.int().cpu().tolist()   # tracking ID
                    clss        = r.boxes.cls.int().cpu().tolist()  # type (car, bus…)
                    confidences = r.boxes.conf.cpu().numpy()        # how sure the model is

                    for box, obj_id, cls, conf in zip(boxes, ids, clss, confidences):

                        label = model.names[cls]

                        # Skip if it is not a vehicle type we care about
                        if label not in VEHICLE_CLASSES or conf < MIN_CONFIDENCE:
                            continue

                        # The vertical centre of the bounding box.
                        # We watch vertical movement because most traffic cameras
                        # are above the road so cars move up or down in the frame.
                        y_center = (box[1] + box[3]) / 2.0

                        # --- SPEED ---
                        # Compare where the vehicle is now to where it was last frame.
                        # pixels moved × metres-per-pixel × frames-per-second = metres/sec
                        # metres/sec × 3.6 = km/h
                        if obj_id in prev_pos:
                            dy        = abs(y_center - prev_pos[obj_id])
                            speed_ms  = dy * PXM * fps
                            speed_kmh = speed_ms * 3.6
                            if speed_kmh > MIN_SPEED_KMH:
                                # Which half of the frame is the vehicle in?
                                bucket = "Inbound" if y_center > h / 2 else "Outbound"
                                current_speeds[bucket].append(speed_kmh)

                        prev_pos[obj_id] = y_center  # remember position for next frame

                        # --- COUNT ---
                        # Only count the first time we see this ID
                        if obj_id not in counted_ids:
                            counted_ids.add(obj_id)
                            entry_points[obj_id] = y_center
                            final_counts[label] += 1

                        # --- DIRECTION ---
                        # Compare current position to where we first saw this vehicle.
                        # Moving down the frame = Inbound.
                        # Moving up the frame   = Outbound.
                        # Once decided, we set a sentinel value so we never count it again.
                        if obj_id in entry_points:
                            ref = entry_points[obj_id]
                            if ref not in (99999, -99999):
                                if y_center > ref + DIRECTION_THRESHOLD:
                                    direction_counts["Inbound"] += 1
                                    entry_points[obj_id] = 99999   # locked — going in
                                elif y_center < ref - DIRECTION_THRESHOLD:
                                    direction_counts["Outbound"] += 1
                                    entry_points[obj_id] = -99999  # locked — going out

                        # --- DRAW BOX ON FRAME ---
                        if st.session_state.config["overlays"]:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                            cv2.putText(
                                frame,
                                f"{label} #{obj_id} {int(conf * 100)}%",
                                (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1,
                            )

                # Save the average speed for this frame into the history lists
                for d in ("Inbound", "Outbound"):
                    if current_speeds[d]:
                        speed_history[d].append(float(np.mean(current_speeds[d])))

                # --- DRAW THE INFO PANEL ON THE RIGHT SIDE OF THE FRAME ---
                # We copy the frame, paint a dark rectangle over the copy,
                # then mix the two together so it looks semi-transparent.
                if st.session_state.config["burn"]:
                    sidebar_w = int(w * HUD_WIDTH_RATIO)
                    x_hud     = w - sidebar_w
                    overlay   = frame.copy()
                    cv2.rectangle(overlay, (x_hud, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

                    # Location name at the top
                    cv2.putText(
                        frame, st.session_state.config["loc"].upper(),
                        (x_hud + 10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1,
                    )

                    # Vehicle counts
                    y_off = 70
                    for obj, val in final_counts.items():
                        cv2.putText(
                            frame, f"{obj.upper()}: {val}",
                            (x_hud + 10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 255), 1,
                        )
                        y_off += 26

                    # Traffic state labels
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

                    # Frame number at the bottom of the panel
                    cv2.putText(
                        frame, f"Frame {i+1}/{max_frames}",
                        (x_hud + 10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1,
                    )

                out.write(frame)  # add this finished frame to the output video

            # --- Wrap up ---
            out.release()
            progress_bar.progress(1.0)
            frame_info.empty()

            # Convert the video to a format that web browsers can play.
            # The raw format we wrote (mp4v) does not always work in browsers.
            # h264 / libx264 works everywhere.
            status.update(label=T["proc_encode"], state="running")
            final_path = _register_tmp(
                tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            )
            clip = VideoFileClip(raw_path)
            clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)
            clip.close()

            status.update(label=T["proc_done"], state="complete")

        # Save everything to session_state so the results section below
        # can read it even after Streamlit re-runs the script
        st.session_state.speed_history    = speed_history
        st.session_state.final_counts     = final_counts
        st.session_state.direction_counts = direction_counts
        st.session_state.unique_count     = len(counted_ids)
        st.session_state.final_path       = final_path

        # =====================================================================
        # RESULTS SCREEN
        # =====================================================================

        st.title(T["results_header"])
        st.divider()

        # Big number cards across the top
        avg_spd_in  = round(np.mean(speed_history["Inbound"]),  1) if speed_history["Inbound"]  else 0.0
        avg_spd_out = round(np.mean(speed_history["Outbound"]), 1) if speed_history["Outbound"] else 0.0
        total       = sum(final_counts.values())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total vehicles | סה״כ רכבים",       total)
        k2.metric("Unique IDs | מזהים ייחודיים",        len(counted_ids))
        k3.metric("Coming in | נכנסים",                 direction_counts["Inbound"])
        k4.metric("Going out | יוצאים",                 direction_counts["Outbound"])
        k5.metric("Avg speed (km/h) | מהירות ממוצעת",  f"{max(avg_spd_in, avg_spd_out):.1f}")

        st.divider()

        # Video on the left, chart and breakdown on the right
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
                # Build a simple table and export it as a CSV file
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

        # Live numbers in the sidebar
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
