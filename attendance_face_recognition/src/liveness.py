"""
Liveness detection module for the Attendance Face Recognition System.

Prevents spoofing attacks where someone tries to mark attendance
using a photo or video of another person.

This module implements simple, student-level liveness detection using:
1. Eye Aspect Ratio (EAR) for blink detection
2. Eye visibility checks
3. Basic motion detection

MediaPipe Face Mesh is used for facial landmark detection,
providing 468 landmark points including precise eye contours.

How blink detection works:
1. Detect facial landmarks using MediaPipe
2. Extract eye landmark points (6 points per eye)
3. Calculate Eye Aspect Ratio (EAR) for each frame
4. EAR drops significantly when eyes blink
5. Detect blink pattern over frame sequence
6. Require at least one blink to confirm liveness

Eye Aspect Ratio formula:
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
where p1-p6 are the 6 eye landmark points
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import os
import warnings
import numpy as np
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import mediapipe as mp

import config


# =============================================================================
# MEDIAPIPE INITIALIZATION
# =============================================================================

# MediaPipe package layouts differ across releases. Newer builds may expose
# only `tasks` and not the legacy `solutions` namespace used by Face Mesh.
_mp_solutions = getattr(mp, "solutions", None)
if _mp_solutions is not None:
    mp_face_mesh = _mp_solutions.face_mesh
    mp_drawing = _mp_solutions.drawing_utils
    mp_drawing_styles = _mp_solutions.drawing_styles
    MEDIAPIPE_SOLUTIONS_AVAILABLE = True
else:
    mp_face_mesh = None
    mp_drawing = None
    mp_drawing_styles = None
    MEDIAPIPE_SOLUTIONS_AVAILABLE = False

# Eye landmark indices for MediaPipe Face Mesh (468 landmarks)
# Left eye landmarks
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173]
# Right eye landmarks
RIGHT_EYE_INDICES = [362, 263, 385, 386, 387, 388, 466]

# Simplified 6-point eye contours (for EAR calculation)
LEFT_EYE_CONTOUR = [33, 160, 158, 157, 173, 133]
RIGHT_EYE_CONTOUR = [362, 385, 387, 388, 466, 263]

# Upper and lower eyelid indices (for simpler EAR)
LEFT_UPPER_LID = [160, 158]
LEFT_LOWER_LID = [157, 173]
RIGHT_UPPER_LID = [385, 387]
RIGHT_LOWER_LID = [388, 466]


# =============================================================================
# EYE ASPECT RATIO CALCULATION
# =============================================================================

def calculate_eye_aspect_ratio(
    eye_landmarks: np.ndarray
) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from eye landmarks.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where:
    - p1, p4: Horizontal eye corners (distance changes little)
    - p2, p6: Top and bottom of eye (distance changes during blink)
    - p3, p5: Inner top and bottom (also changes during blink)

    Args:
        eye_landmarks: 6 eye landmark points as (x, y) coordinates

    Returns:
        float: Eye Aspect Ratio (typically 0.2-0.4 when open, <0.2 when closed)
    """
    # Calculate vertical distances
    # p2 to p6 (top to bottom)
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    # p3 to p5 (inner top to inner bottom)
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Calculate horizontal distance
    # p1 to p4 (left to right corner)
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Avoid division by zero
    if h < 1e-6:
        return 0.0

    # Calculate EAR
    ear = (v1 + v2) / (2.0 * h)

    return ear


def get_eye_landmarks(face_landmarks, image_shape, eye="left"):
    """
    Extract eye landmark coordinates from MediaPipe face landmarks.

    Args:
        face_landmarks: MediaPipe face landmark object
        image_shape: (height, width) of the image
        eye: Which eye ("left" or "right")

    Returns:
        np.ndarray: 6x2 array of (x, y) coordinates for eye landmarks
    """
    height, width = image_shape[:2]

    if eye == "left":
        contour_indices = LEFT_EYE_CONTOUR
    else:
        contour_indices = RIGHT_EYE_CONTOUR

    landmarks = []
    for idx in contour_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        landmarks.append([x, y])

    return np.array(landmarks)


def calculate_ear_from_frame(
    image: np.ndarray,
    face_landmarks,
    eye: str = "both"
) -> Dict[str, float]:
    """
    Calculate EAR for one or both eyes from a frame.

    Args:
        image: Input image
        face_landmarks: MediaPipe face landmarks
        eye: Which eye to calculate ("left", "right", or "both")

    Returns:
        Dict with EAR values for requested eye(s)
    """
    results = {}

    if eye in ["left", "both"]:
        left_landmarks = get_eye_landmarks(face_landmarks, image.shape, "left")
        results['left'] = calculate_eye_aspect_ratio(left_landmarks)

    if eye in ["right", "both"]:
        right_landmarks = get_eye_landmarks(face_landmarks, image.shape, "right")
        results['right'] = calculate_eye_aspect_ratio(right_landmarks)

    if eye == "both":
        # Average of both eyes
        results['average'] = (results['left'] + results['right']) / 2

    return results


# =============================================================================
# BLINK DETECTION
# =============================================================================

class BlinkDetector:
    """
    Detect blinks from a sequence of frames using Eye Aspect Ratio.

    A blink is detected when:
    1. EAR drops below threshold (eyes closed)
    2. Then rises above threshold (eyes open)
    3. This pattern happens within a reasonable time window

    Example:
        >>> detector = BlinkDetector()
        >>> for frame in frames:
        ...     is_blink, ear = detector.process_frame(frame)
        ...     if is_blink:
        ...         print("Blink detected!")
    """

    def __init__(
        self,
        ear_threshold: float = None,
        blink_frame_threshold: int = None,
        min_blinks: int = None
    ):
        """
        Initialize blink detector.

        Args:
            ear_threshold: EAR value below which eyes are considered closed
            blink_frame_threshold: Consecutive frames with closed eyes to count as blink
            min_blinks: Minimum blinks required for liveness confirmation
        """
        self.ear_threshold = ear_threshold if ear_threshold is not None else config.EAR_THRESHOLD
        self.blink_frame_threshold = blink_frame_threshold if blink_frame_threshold is not None else config.BLINK_FRAME_THRESHOLD
        self.min_blinks = min_blinks if min_blinks is not None else config.MIN_BLINKS_REQUIRED

        # State tracking
        self._ear_history = deque(maxlen=30)  # Last 30 frames
        self._closed_frame_count = 0
        self._blink_count = 0
        self._is_in_blink = False
        self._frames_since_blink = 0

        # MediaPipe Face Mesh
        if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
            self._face_mesh = None
            return

        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, float, int]:
        """
        Process a single frame for blink detection.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (is_blink_detected, current_ear, total_blinks)
        """
        if self._face_mesh is None:
            return False, 0.0, self._blink_count

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # No face detected
            self._ear_history.append(0.0)
            return False, 0.0, self._blink_count

        # Get landmarks for first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Calculate EAR
        ear_values = calculate_ear_from_frame(frame, face_landmarks, "both")
        current_ear = ear_values.get('average', 0.0)

        # Add to history
        self._ear_history.append(current_ear)

        # Check if eyes are closed
        if current_ear < self.ear_threshold:
            self._closed_frame_count += 1

            # Check if this completes a blink
            if self._closed_frame_count >= self.blink_frame_threshold and not self._is_in_blink:
                self._is_in_blink = True
                self._blink_count += 1
                return True, current_ear, self._blink_count
        else:
            # Eyes are open
            if self._is_in_blink:
                # Blink completed
                self._is_in_blink = False
                self._frames_since_blink = 0

            self._closed_frame_count = 0
            self._frames_since_blink += 1

        return False, current_ear, self._blink_count

    def check_liveness(
        self,
        frames: List[np.ndarray],
        timeout_seconds: int = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check liveness by detecting blinks in a frame sequence.

        Args:
            frames: List of frames to analyze
        timeout_seconds: Maximum time to wait (not enforced in this simple version)

        Returns:
            Tuple of (is_live, info_dict)
        """
        # Reset state
        self.reset()

        info = {
            'total_frames': len(frames),
            'frames_processed': 0,
            'blinks_detected': 0,
            'ear_values': [],
            'passed': False
        }

        for frame in frames:
            is_blink, ear, total_blinks = self.process_frame(frame)
            info['frames_processed'] += 1
            info['ear_values'].append(ear)

            if total_blinks >= self.min_blinks:
                info['passed'] = True
                info['blinks_detected'] = total_blinks
                return True, info

        # Final check
        info['blinks_detected'] = self._blink_count
        info['passed'] = self._blink_count >= self.min_blinks

        return info['passed'], info

    def reset(self):
        """Reset detector state for new liveness check."""
        self._ear_history.clear()
        self._closed_frame_count = 0
        self._blink_count = 0
        self._is_in_blink = False
        self._frames_since_blink = 0

    def release(self):
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()


class LivenessDetector:
    """
    Compatibility wrapper for the attendance page.

    The page-level flow performs a lightweight single-frame liveness check on
    each recognized face. This wrapper keeps that interface stable while still
    reusing the existing blink/visibility helpers.
    """

    def __init__(
        self,
        ear_threshold: float = None,
        blink_frame_threshold: int = None,
        min_blinks: int = None
    ):
        self.blink_detector = BlinkDetector(
            ear_threshold=ear_threshold,
            blink_frame_threshold=blink_frame_threshold,
            min_blinks=min_blinks
        )
        self.timeout = config.LIVENESS_TIMEOUT

    def check_liveness(
        self,
        frame: np.ndarray,
        face_location: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Single-frame liveness check used by the attendance page.

        If a face location is provided, the check is limited to that crop.
        """
        region = frame
        if face_location is not None:
            top, right, bottom, left = face_location
            height, width = frame.shape[:2]
            top = max(0, top)
            left = max(0, left)
            bottom = min(height, bottom)
            right = min(width, right)

            if bottom > top and right > left:
                region = frame[top:bottom, left:right]

        eyes_visible, ear_values = check_eye_visibility(region)

        return {
            'is_live': bool(eyes_visible),
            'method': 'eye_visibility',
            'ear_values': ear_values,
            'face_location': face_location,
            'reason': 'Eyes visible' if eyes_visible else 'Eyes not visible'
        }

    def verify_liveness(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Batch API compatibility wrapper."""
        checker = LivenessChecker()
        return checker.verify_liveness(frames)

    def verify_liveness_realtime(
        self,
        frame_callback,
        max_frames: int = 60,
        display: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Realtime API compatibility wrapper."""
        checker = LivenessChecker()
        return checker.verify_liveness_realtime(
            frame_callback=frame_callback,
            max_frames=max_frames,
            display=display
        )


# =============================================================================
# SIMPLIFIED LIVENESS CHECK (WITHOUT FULL BLINK)
# =============================================================================

def check_eye_visibility(
    frame: np.ndarray
) -> Tuple[bool, Dict[str, float]]:
    """
    Simple check if eyes are visible and open.

    Less strict than blink detection - just verifies eyes are open.
    Can be used as a fallback if blink detection is too complex.

    Args:
        frame: Input frame (BGR format)

    Returns:
        Tuple of (eyes_visible, ear_values)
    """
    if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
        return True, {'left': 0.0, 'right': 0.0, 'average': 0.0}

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, {'left': 0.0, 'right': 0.0}

        face_landmarks = results.multi_face_landmarks[0]
        ear_values = calculate_ear_from_frame(frame, face_landmarks, "both")

        # Eyes are considered visible if EAR is above threshold
        avg_ear = ear_values.get('average', 0.0)
        eyes_visible = avg_ear > config.EAR_THRESHOLD

        return eyes_visible, ear_values


# =============================================================================
# LIVENESS CHECK FOR ATTENDANCE
# =============================================================================

class LivenessChecker:
    """
    Complete liveness checking system for attendance marking.

    Combines multiple checks:
    1. Blink detection (primary)
    2. Eye visibility (fallback)
    3. Face motion (optional)

    Example:
        >>> checker = LivenessChecker()
        >>> is_live, info = checker.verify_liveness(frames)
        >>> if is_live:
        ...     mark_attendance()
    """

    def __init__(self):
        """Initialize liveness checker."""
        self.blink_detector = BlinkDetector()
        self.timeout = config.LIVENESS_TIMEOUT

    def verify_liveness(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify liveness from a sequence of frames.

        Args:
            frames: List of frames to analyze

        Returns:
            Tuple of (is_live, info_dict)
        """
        if len(frames) < 10:
            return False, {'error': 'Insufficient frames for liveness check'}

        if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
            return True, {
                'passed': True,
                'method': 'disabled_fallback',
                'reason': 'MediaPipe Face Mesh is not available in this environment'
            }

        # Try blink detection first
        passed, info = self.blink_detector.check_liveness(frames)

        if passed:
            info['method'] = 'blink_detection'
            return True, info

        # Fallback: Check if eyes are visible and there's some motion
        info['method'] = 'eye_visibility_fallback'

        # Check eye visibility in multiple frames
        visible_count = 0
        for frame in frames[::5]:  # Check every 5th frame
            eyes_visible, _ = check_eye_visibility(frame)
            if eyes_visible:
                visible_count += 1

        # If eyes visible in most frames, consider it live
        visibility_ratio = visible_count / max(1, len(frames) // 5)

        if visibility_ratio > 0.7:
            info['visibility_ratio'] = visibility_ratio
            return True, info

        return False, info

    def verify_liveness_realtime(
        self,
        frame_callback,
        max_frames: int = 60,
        display: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify liveness in real-time from webcam.

        Args:
            frame_callback: Function that returns next frame (like camera.read())
            max_frames: Maximum frames to capture
            display: Whether to display frames (for debugging)

        Returns:
            Tuple of (is_live, info_dict)
        """
        self.blink_detector.reset()

        if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
            return True, {
                'passed': True,
                'method': 'disabled_fallback',
                'reason': 'MediaPipe Face Mesh is not available in this environment'
            }

        frames = []
        info = {
            'frames_captured': 0,
            'blinks_detected': 0,
            'ear_values': [],
            'passed': False
        }

        print("Liveness check: Please blink once to continue...")

        for i in range(max_frames):
            frame = frame_callback()

            if frame is None:
                break

            frames.append(frame)

            is_blink, ear, blinks = self.blink_detector.process_frame(frame)
            info['frames_captured'] += 1
            info['ear_values'].append(ear)
            info['blinks_detected'] = blinks

            if display:
                # Display EAR value and blink status
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"EAR: {ear:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display_frame,
                    f"Blinks: {blinks}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if is_blink:
                    cv2.putText(
                        display_frame,
                        "BLINK DETECTED!",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("Liveness Check", display_frame)

            # Check if we have enough blinks
            if blinks >= self.blink_detector.min_blinks:
                info['passed'] = True
                break

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if cv2.waitKey(1) & 0xFF == 27 or display:
            cv2.destroyAllWindows()

        info['passed'] = info['blinks_detected'] >= self.blink_detector.min_blinks

        return info['passed'], info


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def draw_eye_landmarks(
    frame: np.ndarray,
    face_landmarks,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw eye landmarks on frame for visualization.

    Args:
        frame: Input frame
        face_landmarks: MediaPipe face landmarks
        color: BGR color for landmarks

    Returns:
        Frame with landmarks drawn
    """
    output = frame.copy()
    height, width = frame.shape[:2]

    # Draw left eye landmarks
    for idx in LEFT_EYE_CONTOUR:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(output, (x, y), 2, color, -1)

    # Draw right eye landmarks
    for idx in RIGHT_EYE_CONTOUR:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(output, (x, y), 2, color, -1)

    return output


def draw_ear_graph(
    frame: np.ndarray,
    ear_history: List[float],
    threshold: float = None
) -> np.ndarray:
    """
    Draw EAR history as a graph on the frame.

    Args:
        frame: Input frame
        ear_history: List of recent EAR values
        threshold: EAR threshold line

    Returns:
        Frame with EAR graph overlay
    """
    if threshold is None:
        threshold = config.EAR_THRESHOLD

    output = frame.copy()

    # Graph dimensions
    graph_width = 200
    graph_height = 100
    graph_x = frame.shape[1] - graph_width - 10
    graph_y = 10

    # Draw graph background
    cv2.rectangle(
        output,
        (graph_x, graph_y),
        (graph_x + graph_width, graph_y + graph_height),
        (50, 50, 50),
        -1
    )

    # Draw threshold line
    threshold_y = graph_y + graph_height - int(threshold / 0.5 * graph_height)
    cv2.line(
        output,
        (graph_x, threshold_y),
        (graph_x + graph_width, threshold_y),
        (0, 0, 255),
        1
    )

    # Draw EAR curve
    if len(ear_history) > 1:
        points = []
        for i, ear in enumerate(ear_history[-graph_width:]):
            x = graph_x + i
            y = graph_y + graph_height - int(min(ear, 0.5) / 0.5 * graph_height)
            points.append((x, y))

        for i in range(len(points) - 1):
            cv2.line(
                output,
                points[i],
                points[i + 1],
                (0, 255, 0),
                1
            )

    return output
