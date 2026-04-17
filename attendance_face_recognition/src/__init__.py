"""
Attendance Face Recognition System - Source Package

Keep package initialization lightweight.

Streamlit Cloud imports package submodules like `src.attendance_manager`.
If this file eagerly imports every module in `src`, optional heavy
dependencies such as OpenCV get imported before the app even reaches the page
that needs them. That can crash cloud startup unnecessarily.
"""

__all__ = [
    "database",
    "utils",
    "preprocess",
    "face_detector",
    "face_encoder",
    "recognizer",
    "liveness",
    "attendance_manager",
    "registration",
    "report_generator",
    "camera",
]

__version__ = "1.0.0"
