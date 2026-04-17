"""
Home page and app entry point.

Run with: streamlit run app.py
"""

import streamlit as st

import config
from src.attendance_manager import AttendanceManager
from src.database import create_tables, get_all_students
from src.recognizer import FaceRecognizer

st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

create_tables()


if "att_manager" not in st.session_state:
    st.session_state.att_manager = AttendanceManager()

if "recognizer" not in st.session_state:
    recognizer = FaceRecognizer()
    recognizer.load_known_encodings()
    st.session_state.recognizer = recognizer


with st.sidebar:
    st.title("Face Attendance")
    st.caption("Local attendance tracker")
    st.markdown("---")

    manager = st.session_state.att_manager
    stats = manager.get_today_stats()
    encoding_count = st.session_state.recognizer.get_encoding_count()
    students = get_all_students()

    st.metric("Registered students", len(students))
    st.metric("Marked today", stats["total_marked"])
    if stats["session_active"]:
        st.info(f"Session active: {stats['current_subject']}")

    st.markdown("---")
    threshold = st.slider(
        "Recognition threshold",
        0.35,
        0.65,
        config.RECOGNITION_THRESHOLD,
        0.01,
        help="Lower values are stricter.",
    )
    if threshold != config.RECOGNITION_THRESHOLD:
        config.RECOGNITION_THRESHOLD = threshold
        st.session_state.recognizer.threshold = threshold

    st.caption(f"Face encodings loaded: {encoding_count}")


st.title("Face Attendance System")
st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        ## Overview

        This system records student attendance with face recognition.

        ### Core functions
        - Register students with multiple face photos
        - Detect duplicate face records during registration
        - Mark attendance in real time from the camera feed
        - Use liveness checks to reduce photo spoofing
        - Review reports and attendance trends
        """
    )

with col_right:
    st.subheader("System Status")
    if config.DATABASE_PATH.exists():
        st.success("Database ready")
    else:
        st.warning("Database will be created on first run")

    if encoding_count:
        st.success(f"{encoding_count} face encodings loaded")
    else:
        st.info("No face encodings loaded yet")

st.markdown("---")
st.subheader("Quick actions")

action_1, action_2, action_3, action_4 = st.columns(4)
if action_1.button("Register student", use_container_width=True):
    st.switch_page("pages/1_Register_Student.py")
if action_2.button("Mark attendance", use_container_width=True):
    st.switch_page("pages/2_Mark_Attendance.py")
if action_3.button("View reports", use_container_width=True):
    st.switch_page("pages/3_View_Reports.py")
if action_4.button("Analytics", use_container_width=True):
    st.switch_page("pages/4_Analytics.py")

st.markdown("---")
st.subheader("Today")

summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric("Date", stats["date"])
summary_2.metric("Marked today", stats["total_marked"])
summary_3.metric("Session", "Active" if stats["session_active"] else "Inactive")
summary_4.metric("Students on file", len(students))
