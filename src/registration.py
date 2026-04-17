"""
Student registration logic.

Flow:
  1. Validate input data
  2. Generate face encodings from captured frames
  3. Check for duplicate face records
  4. Insert the student record and encodings
  5. Save cropped face images to disk
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import cv2
import face_recognition
import numpy as np

import config
from src.database import get_all_encodings, get_student, insert_encoding, insert_student
from src.utils import bytes_to_encoding, encoding_to_bytes, euclidean_distance, safe_name, validate_student_id, validate_student_name

warnings.filterwarnings("ignore", category=UserWarning)


def validate_student_data(student_id: str, name: str, department: str, year: int, section: str) -> Tuple[bool, str]:
    ok, err = validate_student_id(student_id)
    if not ok:
        return False, f"Student ID: {err}"
    ok, err = validate_student_name(name)
    if not ok:
        return False, f"Name: {err}"
    if not department.strip():
        return False, "Department cannot be empty"
    if year not in range(1, 5):
        return False, "Year must be 1-4"
    if not section.strip():
        return False, "Section cannot be empty"
    return True, ""


def _rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _largest_face(locations):
    """Return the largest face location (closest to the camera)."""
    return max(locations, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))


def generate_encodings_from_frames(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    Extract one face encoding per frame.

    Returns (encodings, face_locations) for valid frames.
    """
    encodings, locations = [], []
    for frame in frames:
        rgb = _rgb(frame)
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        if not locs:
            continue
        loc = _largest_face(locs)
        encs = face_recognition.face_encodings(rgb, [loc])
        if encs:
            encodings.append(encs[0])
            locations.append(loc)
    return encodings, locations


def find_duplicate_face(new_encodings: List[np.ndarray], threshold: float = None) -> Optional[Dict[str, Any]]:
    """
    Check whether any of the new encodings already belong to an existing student.
    """
    if threshold is None:
        threshold = config.RECOGNITION_THRESHOLD

    existing = get_all_encodings()
    if not existing or not new_encodings:
        return None

    best: Optional[Dict[str, Any]] = None

    for new_enc in new_encodings:
        for record in existing:
            try:
                stored_enc = bytes_to_encoding(record["encoding"])
            except Exception:
                continue
            dist = euclidean_distance(new_enc, stored_enc)
            if dist < threshold and (best is None or dist < best["distance"]):
                best = {
                    "student_id": record["student_id"],
                    "name": record["name"],
                    "distance": round(dist, 4),
                }

    return best


def register_student(
    student_id: str,
    name: str,
    department: str,
    year: int,
    section: str,
    frames: List[np.ndarray],
    save_images: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Full registration pipeline.
    """
    meta: Dict[str, Any] = {
        "student_id": student_id,
        "frames_captured": len(frames),
        "encodings_stored": 0,
    }

    ok, err = validate_student_data(student_id, name, department, year, section)
    if not ok:
        return False, err, meta

    if get_student(student_id):
        return False, f"Student ID '{student_id}' is already registered.", meta

    if len(frames) < 5:
        return False, f"Too few frames ({len(frames)}). Capture at least 5.", meta

    encodings, face_locs = generate_encodings_from_frames(frames)
    meta["encodings_generated"] = len(encodings)

    if len(encodings) < 3:
        return False, "Could not generate enough face encodings. Try better lighting.", meta

    duplicate = find_duplicate_face(encodings)
    if duplicate:
        return (
            False,
            f"Duplicate face detected.\n"
            f"This face already belongs to '{duplicate['name']}' "
            f"(ID: {duplicate['student_id']}).\n"
            f"Registration blocked.",
            meta,
        )

    if not insert_student(student_id, name, department, year, section):
        return False, "Failed to create student record (database error).", meta

    student_folder = config.FACES_DIR / safe_name(student_id)
    if save_images:
        student_folder.mkdir(parents=True, exist_ok=True)

    keep = min(len(encodings), config.NUM_ENCODINGS_PER_STUDENT)
    for i, (enc, loc) in enumerate(zip(encodings[:keep], face_locs[:keep])):
        image_path = None
        if save_images:
            top, right, bottom, left = loc
            face_crop = _rgb(frames[i])[top:bottom, left:right]
            image_path = str(student_folder / f"face_{i:03d}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        if insert_encoding(student_id, encoding_to_bytes(enc), image_path):
            meta["encodings_stored"] += 1

    if meta["encodings_stored"] < 3:
        return False, "Not enough encodings stored. Try again.", meta

    return True, f"'{name}' registered successfully!", meta


def check_frame_for_registration(frame: np.ndarray) -> Tuple[bool, str, Optional[Tuple]]:
    """
    Quick check: does this frame have exactly one usable face?
    """
    rgb = _rgb(frame)
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)

    if not locs:
        return False, "No face detected. Position your face in the frame.", None
    if len(locs) > 1:
        return False, f"{len(locs)} faces detected. Only one person at a time.", None

    loc = locs[0]
    top, right, bottom, left = loc
    face_area = (bottom - top) * (right - left)
    frame_area = frame.shape[0] * frame.shape[1]
    if face_area / frame_area < config.MIN_FACE_SIZE:
        return False, "Face too small. Move closer to the camera.", None

    face_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(face_gray))
    if brightness < 50:
        return False, "Too dark. Improve the lighting.", None
    if brightness > 220:
        return False, "Overexposed. Reduce lighting.", None

    return True, "Ready", loc
