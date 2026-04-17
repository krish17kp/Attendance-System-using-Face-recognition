"""
pages/1_Register_Student.py

Student registration flow:
  Step 1 – Fill in the form.
  Step 2 – Capture face photos via browser camera.
  Step 3 – System generates encodings and checks for duplicate faces.
  Step 4 – Registration complete (or error shown).
"""

import cv2
import numpy as np
import streamlit as st

import config
from src.database import create_tables, get_student
from src.registration import check_frame_for_registration, register_student, validate_student_data

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Register Student", page_icon="📝", layout="wide",
                   initial_sidebar_state="collapsed")

create_tables()

# ── Session state defaults ────────────────────────────────────────────────────
for key, val in [
    ("student_data", None),
    ("captured_frames", []),
    ("last_bytes", None),
    ("reg_complete", False),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_camera(file) -> np.ndarray | None:
    if file is None:
        return None
    buf = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def reset(clear_student: bool = True):
    st.session_state.captured_frames = []
    st.session_state.last_bytes      = None
    if clear_student:
        st.session_state.student_data = None


# ── Registration form ─────────────────────────────────────────────────────────

def render_form():
    st.subheader("Student Details")
    existing = st.session_state.student_data or {}

    with st.form("reg_form"):
        c1, c2 = st.columns(2)
        with c1:
            sid  = st.text_input("Student ID *", value=existing.get("student_id", ""),
                                  placeholder="e.g. 22BCE001")
            name = st.text_input("Full Name *",  value=existing.get("name", ""),
                                  placeholder="e.g. Aanya Patel")
            dept = st.selectbox("Department *", config.DEPARTMENTS,
                                 index=config.DEPARTMENTS.index(
                                     existing.get("department", config.DEPARTMENTS[0])))
        with c2:
            year = st.selectbox("Year *", config.YEARS,
                                 index=config.YEARS.index(existing.get("year", 1)),
                                 format_func=lambda y: f"Year {y}")
            sec  = st.selectbox("Section *", config.SECTIONS,
                                 index=config.SECTIONS.index(existing.get("section", "A")))
        submitted = st.form_submit_button("▶ Start Face Capture", type="primary")

    if not submitted:
        return

    ok, err = validate_student_data(sid, name, dept, year, sec)
    if not ok:
        st.error(err); return

    if get_student(sid):
        st.error(f"Student ID '{sid}' is already registered."); return

    st.session_state.student_data = dict(
        student_id=sid, name=name, department=dept, year=year, section=sec
    )
    st.rerun()


# ── Camera capture ────────────────────────────────────────────────────────────

def render_capture(student: dict):
    st.subheader("Face Capture")
    st.info(f"Registering **{student['name']}** ({student['student_id']})")

    n = len(st.session_state.captured_frames)
    st.progress(min(1.0, n / config.NUM_REGISTRATION_IMAGES))
    st.caption(f"{n} / {config.NUM_REGISTRATION_IMAGES} photos saved  "
               f"(at least 5 required to finish)")

    img_file = st.camera_input("Take a photo", key="reg_cam")
    frame    = decode_camera(img_file)

    valid, msg, _ = (check_frame_for_registration(frame)
                     if frame is not None else (False, "Take a photo first.", None))
    if frame is not None:
        (st.success if valid else st.warning)(msg)

    c1, c2, c3, c4 = st.columns(4)

    if c1.button("💾 Save Photo", type="primary"):
        if frame is None:
            st.error("Take a photo first.")
        elif not valid:
            st.error(msg)
        elif img_file.getvalue() == st.session_state.last_bytes:
            st.warning("Already saved – take a new photo first.")
        else:
            st.session_state.captured_frames.append(frame.copy())
            st.session_state.last_bytes = img_file.getvalue()
            st.rerun()

    if c2.button("🗑 Clear Photos"):
        reset(clear_student=False); st.rerun()

    if c3.button("✏️ Edit Details"):
        reset(clear_student=True); st.rerun()

    if c4.button("✅ Finish Registration"):
        if n < 5:
            st.error("Capture at least 5 photos before finishing.")
        else:
            return True          # Signal: proceed to processing
    return False


# ── Processing ────────────────────────────────────────────────────────────────

def process_registration(student: dict, frames: list):
    with st.spinner("Generating face encodings and checking for duplicates…"):
        ok, msg, meta = register_student(
            student_id  = student["student_id"],
            name        = student["name"],
            department  = student["department"],
            year        = student["year"],
            section     = student["section"],
            frames      = frames,
        )

    if ok:
        st.success(msg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames captured",    meta["frames_captured"])
        c2.metric("Encodings generated", meta.get("encodings_generated", "—"))
        c3.metric("Encodings stored",    meta["encodings_stored"])

        # Reload encodings in the recognizer (if it's already in session)
        if "recognizer" in st.session_state:
            st.session_state.recognizer.load_known_encodings(force=True)

        st.session_state.reg_complete = True
    else:
        st.error(msg)
        # Duplicate face error gets a prominent warning
        if "Duplicate face" in msg or "already belongs" in msg:
            st.warning("⚠️ Registration blocked to prevent duplicate entries.")
        if st.button("Try Again"):
            reset(clear_student=False); st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📝 Register Student")
    st.markdown("---")

    if st.session_state.reg_complete:
        st.success("🎉 Registration complete!")
        if st.button("Register Another Student"):
            st.session_state.reg_complete = False
            reset(clear_student=True)
            st.rerun()
        return

    if st.session_state.student_data is None:
        render_form()
        st.markdown("""
        ---
        **Tips for best results:**
        - Good, even lighting on your face
        - Face the camera directly
        - Try different slight angles across multiple photos
        - Remove glasses if possible
        """)
        return

    s = st.session_state.student_data
    st.markdown(
        f"**ID:** {s['student_id']}  |  **Name:** {s['name']}  |  "
        f"**Dept:** {s['department']}  |  **Year:** {s['year']}  |  **Section:** {s['section']}"
    )
    st.markdown("---")

    should_process = render_capture(s)
    if should_process:
        st.markdown("---")
        process_registration(s, st.session_state.captured_frames)


if __name__ == "__main__":
    main()