"""
recognizer.py — match a webcam face against stored encodings.

How it works:
  1. Load all encodings from DB into memory (cached).
  2. For each frame, generate a 128-D encoding.
  3. Find the minimum Euclidean distance to any stored encoding.
  4. If distance < threshold → recognised; otherwise → Unknown.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import face_recognition
import numpy as np

import config
from src.database import get_all_encodings
from src.utils import bytes_to_encoding, euclidean_distance


class FaceRecognizer:
    """Loads known encodings and matches faces from webcam frames."""

    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.RECOGNITION_THRESHOLD
        self._encodings: List[np.ndarray] = []
        self._ids:       List[str]        = []
        self._names:     List[str]        = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_known_encodings(self, force: bool = False) -> int:
        """Load (or reload) encodings from the database."""
        if self._encodings and not force:
            return len(self._encodings)

        self._encodings, self._ids, self._names = [], [], []
        for rec in get_all_encodings():
            try:
                enc = bytes_to_encoding(rec["encoding"])
                if enc.shape == (128,):
                    self._encodings.append(enc)
                    self._ids.append(rec["student_id"])
                    self._names.append(rec["name"])
            except Exception:
                continue
        return len(self._encodings)

    def get_encoding_count(self) -> int:
        return len(self._encodings)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        frame: np.ndarray,
        face_location: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        """
        Identify a face in *frame*.

        Args:
            frame: BGR webcam frame.
            face_location: (top, right, bottom, left) or None (auto-detect).

        Returns dict with keys:
            recognized (bool), student_id, name, distance, confidence
        """
        result: Dict[str, Any] = {
            "recognized": False, "student_id": None,
            "name": None, "distance": float("inf"), "confidence": 0.0,
        }

        if not self._encodings:
            self.load_known_encodings()
        if not self._encodings:
            return result

        # Convert BGR → RGB
        rgb = frame[:, :, ::-1].copy()

        # Get encoding for the detected face
        locs = [face_location] if face_location else face_recognition.face_locations(rgb)
        if not locs:
            return result
        encs = face_recognition.face_encodings(rgb, [locs[0]])
        if not encs:
            return result

        query_enc = encs[0]

        # Compare against every stored encoding
        distances = [euclidean_distance(query_enc, k) for k in self._encodings]
        best_idx  = int(np.argmin(distances))
        best_dist = distances[best_idx]

        result["distance"] = round(best_dist, 4)

        if best_dist < self.threshold:
            result["recognized"]  = True
            result["student_id"]  = self._ids[best_idx]
            result["name"]        = self._names[best_idx]
            result["confidence"]  = round(max(0.0, 1.0 - best_dist / self.threshold), 3)

        return result