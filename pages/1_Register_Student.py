"""
Student registration page.

Step 1: Fill in the form.
Step 2: Capture face photos from the camera.
Step 3: Save the student record and face encodings.
"""

import cv2
import numpy as np
import streamlit as st

import config
from src.database import create_tables, get_student
from src.registration import check_frame_for_registration, register_student, validate_student_data

st.set_page_config(
    page_title="Register Student",
    layout="wide",
    initial_sidebar_state="collapsed",
)

create_tables()


for key, value in [
    ("student_data", None),
    ("captured_frames", []),
    ("last_bytes", None),
    ("reg_complete", False),
]:
    if key not in st.session_state:
        st.session_state[key] = value


def decode_camera(file) -> np.ndarray | None:
    if file is None:
        return None
    buf = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def reset(clear_student: bool = True):
    st.session_state.captured_frames = []
    st.session_state.last_bytes = None
    if clear_student:
        st.session_state.student_data = None


def render_form():
    st.subheader("Student details")
    existing = st.session_state.student_data or {}

    with st.form("reg_form"):
        c1, c2 = st.columns(2)
        with c1:
            sid = st.text_input("Student ID *", value=existing.get("student_id", ""), placeholder="e.g. 22BCE001")
            name = st.text_input("Full name *", value=existing.get("name", ""), placeholder="e.g. Aanya Patel")
            dept = st.selectbox(
                "Department *",
                config.DEPARTMENTS,
                index=config.DEPARTMENTS.index(existing.get("department", config.DEPARTMENTS[0])),
            )
        with c2:
            year = st.selectbox(
                "Year *",
                config.YEARS,
                index=config.YEARS.index(existing.get("year", 1)),
                format_func=lambda y: f"Year {y}",
            )
            sec = st.selectbox(
                "Section *",
                config.SECTIONS,
                index=config.SECTIONS.index(existing.get("section", "A")),
            )
        submitted = st.form_submit_button("Start face capture", type="primary")

    if not submitted:
        return

    ok, err = validate_student_data(sid, name, dept, year, sec)
    if not ok:
        st.error(err)
        return

    if get_student(sid):
        st.error(f"Student ID '{sid}' is already registered.")
        return

    st.session_state.student_data = dict(
        student_id=sid,
        name=name,
        department=dept,
        year=year,
        section=sec,
    )
    st.rerun()


def render_capture(student: dict):
    st.subheader("Face capture")
    st.info(f"Registering {student['name']} ({student['student_id']})")

    count = len(st.session_state.captured_frames)
    st.progress(min(1.0, count / config.NUM_REGISTRATION_IMAGES))
    st.caption(f"{count} of {config.NUM_REGISTRATION_IMAGES} photos saved. At least 5 are required.")

    img_file = st.camera_input("Capture a photo", key="reg_cam")
    frame = decode_camera(img_file)

    valid, msg, _ = (
        check_frame_for_registration(frame)
        if frame is not None
        else (False, "Capture a photo first.", None)
    )
    if frame is not None:
        (st.success if valid else st.warning)(msg)

    c1, c2, c3, c4 = st.columns(4)

    if c1.button("Save photo", type="primary"):
        if frame is None:
            st.error("Capture a photo first.")
        elif not valid:
            st.error(msg)
        elif img_file.getvalue() == st.session_state.last_bytes:
            st.warning("This photo is already saved. Capture a new one.")
        else:
            st.session_state.captured_frames.append(frame.copy())
            st.session_state.last_bytes = img_file.getvalue()
            st.rerun()

    if c2.button("Clear photos"):
        reset(clear_student=False)
        st.rerun()

    if c3.button("Edit details"):
        reset(clear_student=True)
        st.rerun()

    if c4.button("Finish registration"):
        if count < 5:
            st.error("Capture at least 5 photos before finishing.")
        else:
            return True

    return False


def process_registration(student: dict, frames: list):
    with st.spinner("Processing registration"):
        ok, msg, meta = register_student(
            student_id=student["student_id"],
            name=student["name"],
            department=student["department"],
            year=student["year"],
            section=student["section"],
            frames=frames,
        )

    if ok:
        st.success(msg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames captured", meta["frames_captured"])
        c2.metric("Encodings generated", meta.get("encodings_generated", "—"))
        c3.metric("Encodings stored", meta["encodings_stored"])

        if "recognizer" in st.session_state:
            st.session_state.recognizer.load_known_encodings(force=True)

        st.session_state.reg_complete = True
    else:
        st.error(msg)
        if "Duplicate face" in msg or "already belongs" in msg:
            st.warning("Registration blocked to avoid duplicate records.")
        if st.button("Try again"):
            reset(clear_student=False)
            st.rerun()


def main():
    st.title("Register Student")
    st.markdown("---")

    if st.session_state.reg_complete:
        st.success("Registration complete")
        if st.button("Register another student"):
            st.session_state.reg_complete = False
            reset(clear_student=True)
            st.rerun()
        return

    if st.session_state.student_data is None:
        render_form()
        st.markdown(
            """
            ---
            **Capture tips**
            - Use even lighting
            - Face the camera directly
            - Save a few slight angle changes
            - Keep one person in frame
            """
        )
        return

    student = st.session_state.student_data
    st.markdown(
        f"**ID:** {student['student_id']}  |  **Name:** {student['name']}  |  "
        f"**Dept:** {student['department']}  |  **Year:** {student['year']}  |  **Section:** {student['section']}"
    )
    st.markdown("---")

    if render_capture(student):
        st.markdown("---")
        process_registration(student, st.session_state.captured_frames)


if __name__ == "__main__":
    main()
