"""
app.py — Home page and app entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

import config
from src.attendance_manager import AttendanceManager
from src.database import create_tables, get_all_students
from src.recognizer import FaceRecognizer

st.set_page_config(page_title=config.APP_TITLE, page_icon="📸",
                   layout="wide", initial_sidebar_state="expanded")

create_tables()


# ── Session state ─────────────────────────────────────────────────────────────
if "att_manager" not in st.session_state:
    st.session_state.att_manager = AttendanceManager()

if "recognizer" not in st.session_state:
    r = FaceRecognizer()
    r.load_known_encodings()
    st.session_state.recognizer = r


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Face Attendance")
    st.markdown("---")

    mgr    = st.session_state.att_manager
    stats  = mgr.get_today_stats()
    enc    = st.session_state.recognizer.get_encoding_count()
    stds   = get_all_students()

    st.metric("Registered students", len(stds))
    st.metric("Marked today",        stats["total_marked"])
    if stats["session_active"]:
        st.info(f"Active: {stats['current_subject']}")

    st.markdown("---")
    threshold = st.slider("Recognition threshold", 0.35, 0.65,
                           config.RECOGNITION_THRESHOLD, 0.01,
                           help="Lower = stricter, Higher = more lenient")
    if threshold != config.RECOGNITION_THRESHOLD:
        config.RECOGNITION_THRESHOLD = threshold
        st.session_state.recognizer.threshold = threshold

    st.caption(f"Encodings loaded: {enc}")


# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📸 Face Attendance System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome

    This system uses **facial recognition** to automate classroom attendance.

    ### Features
    - **Register** students with their face (15 photos captured via browser camera)
    - **Duplicate face detection** – blocks re-registration under a different name
    - **Mark attendance** in real time via webcam + face recognition
    - **Liveness detection** (MediaPipe) to prevent photo spoofing
    - **Reports** – filter by date, subject, section, student; export CSV / Excel
    - **Analytics** – trends, department breakdowns, top performers
    """)

with col2:
    st.markdown("### System Status")
    db_ok = config.DATABASE_PATH.exists()
    st.success("✅ Database ready") if db_ok else st.warning("Database will be created on first run")
    enc = st.session_state.recognizer.get_encoding_count()
    if enc:
        st.success(f"✅ {enc} face encodings loaded")
    else:
        st.info("No face encodings – register students first")

st.markdown("---")
st.subheader("Quick Actions")

c1, c2, c3, c4 = st.columns(4)
if c1.button("📝 Register Student",  use_container_width=True):
    st.switch_page("pages/1_Register_Student.py")
if c2.button("📸 Mark Attendance",   use_container_width=True):
    st.switch_page("pages/2_Mark_Attendance.py")
if c3.button("📊 View Reports",      use_container_width=True):
    st.switch_page("pages/3_View_Reports.py")
if c4.button("📈 Analytics",         use_container_width=True):
    st.switch_page("pages/4_Analytics.py")

st.markdown("---")
st.subheader("Today's Summary")
stats = st.session_state.att_manager.get_today_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Date",            stats["date"])
c2.metric("Marked today",    stats["total_marked"])
c3.metric("Session",         "Active" if stats["session_active"] else "Inactive")
c4.metric("Students on file", len(get_all_students()))