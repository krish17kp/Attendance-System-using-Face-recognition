"""
utils.py — small helper functions used across the system.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


# Encoding serialisation
def encoding_to_bytes(enc: np.ndarray) -> bytes:
    return pickle.dumps(enc)

def bytes_to_encoding(data: bytes) -> np.ndarray:
    return pickle.loads(data)


# Distance
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Date / time
def get_current_date() -> str:
    return datetime.now().strftime(config.DATE_FORMAT)

def get_current_time() -> str:
    return datetime.now().strftime(config.TIME_FORMAT)

def format_date_display(date_str: str) -> str:
    return datetime.strptime(date_str, config.DATE_FORMAT).strftime("%d %b %Y")


# Image quality
def image_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Filename helpers
def safe_name(s: str) -> str:
    for ch in r'<>:"/\|?*':
        s = s.replace(ch, "_")
    return s.strip()


# Validation helpers
def validate_student_id(sid: str) -> Tuple[bool, str]:
    sid = sid.strip()
    if not sid:               return False, "ID cannot be empty"
    if len(sid) < 3:          return False, "ID too short (min 3 chars)"
    if len(sid) > 20:         return False, "ID too long (max 20 chars)"
    return True, ""

def validate_student_name(name: str) -> Tuple[bool, str]:
    name = name.strip()
    if not name:              return False, "Name cannot be empty"
    if len(name) < 2:         return False, "Name too short"
    if len(name) > 100:       return False, "Name too long"
    return True, ""