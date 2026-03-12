import streamlit as st
import tempfile
import os
from ultralytics import YOLO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Tactical AI Command", page_icon="🛰️")

# --- SESSION STATE INITIALIZATION ---
# This keeps track of our Yes/No flow without resetting the app
if 'mission_status' not in st.session_state:
    st.session_state.mission_status = 'standby'
if 'total_objects' not in st.session_state:
    st.session_state.total_objects = 0

st.title("🛰️ Tactical AI Command Center")
st.markdown("### Stage 1: Manual Intel Upload")
st.info("Automated extraction bypassed. Awaiting manual file upload.")

# --- STAGE 1: MANUAL UPLOAD ---
uploaded_file = st.file_uploader("Upload Target Video (MP4, MOV, AVI)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # If the user uploads a new file, reset the mission status
    if st.session_state.get('last_file') != uploaded_file.name:
        st.session_state.mission_status = 'standby'
        st.session_state.last_file = uploaded_file.name

    # Display the uploaded video in the cell
    st.video(uploaded_file)
    
    st.markdown("### Stage 2: Tactical Decision")
    
    # --- FLOW: STANDBY ---
    if st.session_state.mission_status == 'standby':
        st.warning("Visual uplink secured. Do you want to proceed with AI analysis?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes (Proceed)", use_container_width=True):
                st.session_state.mission_status = 'analyzing'
                st.rerun() # Instantly updates the UI
        with col2:
            if st.button("❌ No (Abort)", use_container_width=True):
                st.session_state.mission_status = 'aborted'
                st.rerun()

    # --- FLOW: ABORTED ---
    elif st.session_state.mission_status == 'aborted':
        st.error("Mission scrubbed. Standing by for new orders.")
        if st.button("Reset Mission"):
            st.session_state.mission_status = 'standby'
            st.rerun()

    # --- FLOW: ANALYZING ---
    elif st.session_state.mission_status == 'analyzing':
        st.info("AI analysis engaged. Scanning sector...")
        
        # Streamlit requires saving the upload to a temp file so YOLO can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        try:
            with st.spinner("Processing tactical data..."):
                model = YOLO("yolo11n.pt")
                # Using half-precision and smaller image size for speed
                results = model.predict(source=video_path, half=True, imgsz=320)
                
                # Calculate total objects across all frames
                total_objects = sum([len(r.boxes) for r in results])
                
                st.session_state.total_objects = total_objects
                st.session_state.mission_status = 'complete'
                
        except Exception as e:
            st.error(f"Analysis Failed: {e}")
            st.session_state.mission_status = 'failed'
            
        finally:
            # Clean up the temporary file
            if os.path.exists(video_path):
                os.remove(video_path)
                
        st.rerun()

    # --- FLOW: COMPLETE ---
    elif st.session_state.mission_status == 'complete':
        st.success("Mission Complete. Sector scanned.")
        
        # Display Results directly in the UI cell
        st.markdown("### 📊 Tactical Report")
        st.metric(label="Total Objects Detected", value=st.session_state.total_objects)
        
        # Another Yes/No prompt to continue the loop
        st.warning("Do you want to run another scan on this file?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes (Rescan)", use_container_width=True):
                st.session_state.mission_status = 'analyzing'
                st.rerun()
        with col2:
            if st.button("❌ No (Clear)", use_container_width=True):
                st.session_state.mission_status = 'standby'
                st.rerun()
