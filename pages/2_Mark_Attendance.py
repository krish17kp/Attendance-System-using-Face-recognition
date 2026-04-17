"""
Live attendance page.

Features:
  - Subject / section / date session setup
  - Camera loop inside Streamlit
  - Optional liveness check
  - Recent marks panel
"""

import time
from collections import deque
from datetime import datetime

import cv2
import streamlit as st

import config
from src.attendance_manager import AttendanceManager
from src.database import create_tables, get_all_students
from src.recognizer import FaceRecognizer

try:
    from src.liveness import LivenessDetector, MEDIAPIPE_SOLUTIONS_AVAILABLE
except Exception:
    MEDIAPIPE_SOLUTIONS_AVAILABLE = False
    LivenessDetector = None

st.set_page_config(
    page_title="Mark Attendance",
    layout="wide",
    initial_sidebar_state="collapsed",
)

create_tables()


def _init():
    defaults = {
        "att_manager": AttendanceManager(),
        "recognizer": None,
        "liveness": None,
        "session_active": False,
        "cur_subject": config.DEFAULT_SUBJECTS[0],
        "cur_section": "A",
        "recent_marks": deque(maxlen=10),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.recognizer is None:
        recognizer = FaceRecognizer()
        recognizer.load_known_encodings()
        st.session_state.recognizer = recognizer

    if st.session_state.liveness is None and MEDIAPIPE_SOLUTIONS_AVAILABLE:
        st.session_state.liveness = LivenessDetector()


_init()


def render_session_controls() -> bool:
    manager = st.session_state.att_manager

    if not st.session_state.session_active:
        st.subheader("Start session")
        c1, c2, c3 = st.columns(3)
        subject = c1.selectbox("Subject", config.DEFAULT_SUBJECTS, key="att_subj")
        section = c2.selectbox("Section", config.SECTIONS, key="att_sec")
        date = c3.date_input("Date", value=datetime.now(), key="att_date")

        students = get_all_students()
        if not students:
            st.warning("No students are registered yet.")
            if st.button("Go to registration"):
                st.switch_page("pages/1_Register_Student.py")
            return False

        enc_count = st.session_state.recognizer.get_encoding_count()
        st.info(f"{len(students)} students registered, {enc_count} encodings loaded")

        if st.button("Start session", type="primary"):
            ok, _ = manager.start_session(subject, section, date.strftime(config.DATE_FORMAT))
            if ok:
                st.session_state.session_active = True
                st.session_state.cur_subject = subject
                st.session_state.cur_section = section
                st.rerun()
        return False

    st.success("Session active")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subject", st.session_state.cur_subject)
    c2.metric("Section", st.session_state.cur_section)
    c3.metric("Marked today", manager.get_today_stats()["total_marked"])
    if c4.button("End session"):
        manager.end_session()
        st.session_state.session_active = False
        st.rerun()
    return True


def render_live_recognition():
    st.subheader("Live recognition")
    manager = st.session_state.att_manager
    recognizer = st.session_state.recognizer
    liveness = st.session_state.liveness

    use_liveness = st.checkbox(
        "Enable liveness check",
        value=MEDIAPIPE_SOLUTIONS_AVAILABLE,
        disabled=not MEDIAPIPE_SOLUTIONS_AVAILABLE,
    )
    if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
        st.caption("MediaPipe is not available, so liveness checks are disabled.")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    start = st.button("Start recognition", type="primary")
    stop = st.button("Stop")

    if not start:
        return

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        st.error("Cannot open camera.")
        return

    for _ in range(5):
        cap.read()

    last_recognition = datetime.now()
    cooldown_secs = 2.0
    current_result = None
    result_timeout = 3.0

    status_placeholder.info("Looking for a face")

    try:
        while not stop:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            small = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_FACTOR, fy=config.FRAME_RESIZE_FACTOR)
            import face_recognition as fr

            locs_small = fr.face_locations(small[:, :, ::-1])
            scale = 1.0 / config.FRAME_RESIZE_FACTOR
            locs = [(int(t * scale), int(r * scale), int(b * scale), int(l * scale)) for t, r, b, l in locs_small]

            display = frame.copy()
            now = datetime.now()

            if locs:
                elapsed = (now - last_recognition).total_seconds()
                if elapsed >= cooldown_secs:
                    result = recognizer.recognize(frame, locs[0])
                    last_recognition = now

                    if result["recognized"]:
                        sid, name = result["student_id"], result["name"]
                        live_ok = True
                        if use_liveness and liveness:
                            lr = liveness.check_liveness(frame, locs[0])
                            live_ok = lr.get("is_live", False)

                        if live_ok:
                            ok, _, _ = manager.mark_attendance(sid, name)
                            if ok:
                                current_result = {"status": "success", "name": name, "time": now}
                                st.session_state.recent_marks.appendleft(
                                    {"name": name, "id": sid, "time": now.strftime("%H:%M:%S")}
                                )
                            else:
                                current_result = {"status": "duplicate", "name": name, "time": now}
                        else:
                            current_result = {"status": "liveness_fail", "time": now}
                    else:
                        current_result = {"status": "unknown", "time": now}

                top, right, bottom, left = locs[0]
                status = current_result.get("status") if current_result else None
                age = (now - current_result["time"]).total_seconds() if current_result else 999

                if status == "success" and age < result_timeout:
                    color, label = (0, 200, 0), f"{current_result['name']} - marked"
                elif status == "duplicate" and age < result_timeout:
                    color, label = (255, 140, 0), f"{current_result['name']} - already marked"
                elif status == "liveness_fail" and age < result_timeout:
                    color, label = (0, 0, 220), "Liveness check failed"
                elif status == "unknown" and age < result_timeout:
                    color, label = (128, 128, 128), "Unknown face"
                else:
                    color, label = (180, 50, 50), "Detecting"

                cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                cv2.putText(display, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_placeholder.image(display[:, :, ::-1], channels="RGB", use_container_width=True)

            if current_result:
                status = current_result["status"]
                if status == "success":
                    status_placeholder.success(f"Marked: {current_result['name']}")
                elif status == "duplicate":
                    status_placeholder.warning(f"{current_result['name']} is already marked today")
                elif status == "liveness_fail":
                    status_placeholder.error("Liveness check failed")

            time.sleep(0.03)

    finally:
        cap.release()
        frame_placeholder.empty()
        status_placeholder.info("Recognition stopped.")


def render_recent_marks():
    st.subheader("Recent marks")
    marks = list(st.session_state.recent_marks)
    if not marks:
        st.info("No attendance has been marked yet.")
        return
    for mark in marks:
        st.markdown(f"**{mark['name']}** ({mark['id']})  \n<small>{mark['time']}</small>", unsafe_allow_html=True)


def main():
    st.title("Mark Attendance")
    st.markdown("---")

    if not render_session_controls():
        st.markdown(
            """
            ---
            **How it works**
            1. Select the subject, section, and date
            2. Start the session
            3. Start recognition to open the camera
            4. End the session when finished
            """
        )
        return

    st.markdown("---")
    col_cam, col_info = st.columns([2, 1])
    with col_cam:
        render_live_recognition()
    with col_info:
        render_recent_marks()
        st.markdown("---")
        stats = st.session_state.att_manager.get_today_stats()
        st.metric("Total students", len(get_all_students()))
        st.metric("Marked today", stats["total_marked"])
        st.metric("In this session", stats["marked_in_session"])


if __name__ == "__main__":
    main()
