"""
Face detection module for the Attendance Face Recognition System.

Handles all face detection operations using the face_recognition library
(which uses dlib's HOG + HOG-SVM detector internally).

Provides:
- Single face detection
- Multiple face detection
- Face location extraction
- Face count validation

The face_recognition library provides:
- Fast CPU-based detection (no GPU required)
- Good accuracy for frontal and near-frontal faces
- Returns face locations as (top, right, bottom, left) tuples
"""

from typing import List, Tuple, Optional, Union
import warnings
import numpy as np
import cv2

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# face_recognition library uses dlib internally
import face_recognition

import config
from .preprocess import preprocess_frame, resize_frame, convert_bgr_to_rgb


# =============================================================================
# CORE FACE DETECTION
# =============================================================================

def detect_faces(
    image: np.ndarray,
    model: str = "hog",
    num_jitters: int = 1
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image.

    Args:
        image: Input image (RGB format recommended)
        model: Detection model to use
               - "hog": Faster, works on CPU, good for frontal faces
               - "cnn": More accurate, requires GPU, works on various angles
        num_jitters: How many times to resample before detecting
                    Higher = more accurate but slower

    Returns:
        List of face locations as (top, right, bottom, left) tuples
        Empty list if no faces detected

    Example:
        >>> image = cv2.imread('photo.jpg')
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> faces = detect_faces(image_rgb)
        >>> print(f"Found {len(faces)} face(s)")
    """
    # Use face_recognition's built-in detection
    face_locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=num_jitters,
        model=model
    )

    return face_locations


def detect_faces_in_frame(
    frame: np.ndarray,
    resize_factor: Optional[float] = None,
    model: str = "hog"
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a video frame (from webcam).

    This function handles preprocessing (resize + color conversion)
    automatically for optimal performance.

    Args:
        frame: Input frame from webcam (BGR format)
        resize_factor: Factor to resize frame for faster processing
                      (uses config default if None)
        model: Detection model ("hog" or "cnn")

    Returns:
        List of face locations in ORIGINAL frame coordinates
        (scaled back if frame was resized for detection)

    Example:
        >>> cap = cv2.VideoCapture(0)
        >>> ret, frame = cap.read()
        >>> faces = detect_faces_in_frame(frame, resize_factor=0.5)
    """
    if resize_factor is None:
        resize_factor = config.FRAME_RESIZE_FACTOR

    # Preprocess: resize and convert to RGB
    small_frame = preprocess_frame(
        frame,
        resize=True,
        convert_color=True,
        scale_factor=resize_factor
    )

    # Detect faces in the small frame
    face_locations_small = face_recognition.face_locations(
        small_frame,
        number_of_times_to_upsample=1,
        model=model
    )

    # Scale face locations back to original frame size
    if resize_factor < 1.0:
        scale_inv = 1.0 / resize_factor
        face_locations = [
            (
                int(top * scale_inv),
                int(right * scale_inv),
                int(bottom * scale_inv),
                int(left * scale_inv)
            )
            for (top, right, bottom, left) in face_locations_small
        ]
    else:
        face_locations = face_locations_small

    return face_locations


def detect_single_face(
    image: np.ndarray,
    model: str = "hog"
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect exactly one face in an image.

    Useful for registration where we expect one person at a time.

    Args:
        image: Input image (RGB format)
        model: Detection model

    Returns:
        Face location tuple if exactly one face found, None otherwise
    """
    faces = detect_faces(image, model=model)

    if len(faces) == 1:
        return faces[0]
    else:
        return None


def detect_primary_face(
    image: np.ndarray,
    model: str = "hog"
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the primary (largest/most prominent) face in an image.

    When multiple faces are detected, returns the largest one
    (assumed to be closest to camera).

    Args:
        image: Input image (RGB format)
        model: Detection model

    Returns:
        Face location of largest face, None if no faces
    """
    faces = detect_faces(image, model=model)

    if not faces:
        return None

    # Find the largest face by area
    largest_face = max(
        faces,
        key=lambda f: (f[2] - f[0]) * (f[1] - f[3])  # height * width
    )

    return largest_face


# =============================================================================
# FACE DETECTION VALIDATION
# =============================================================================

def count_faces(
    image: np.ndarray,
    model: str = "hog"
) -> int:
    """
    Count number of faces in an image.

    Args:
        image: Input image (RGB format)
        model: Detection model

    Returns:
        Number of faces detected
    """
    return len(detect_faces(image, model=model))


def validate_single_face(
    image: np.ndarray,
    model: str = "hog"
) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]]]:
    """
    Validate that exactly one face is present in an image.

    Returns detailed status for user feedback.

    Args:
        image: Input image (RGB format)
        model: Detection model

    Returns:
        Tuple of (is_valid, message, face_location_or_None)

    Example:
        >>> valid, msg, loc = validate_single_face(image)
        >>> if not valid:
        ...     print(f"Error: {msg}")
    """
    faces = detect_faces(image, model=model)

    if len(faces) == 0:
        return False, "No face detected. Please position your face in the frame.", None

    if len(faces) > 1:
        return False, f"Multiple faces detected ({len(faces)}). Please ensure only one person is in the frame.", None

    return True, "Single face detected successfully.", faces[0]


def validate_face_quality(
    image: np.ndarray,
    face_location: Tuple[int, int, int, int],
    min_size_ratio: float = None
) -> Tuple[bool, str]:
    """
    Validate that a detected face is of sufficient quality.

    Checks:
    - Face is not too small in the frame
    - Face is reasonably well-lit

    Args:
        image: Input image
        face_location: (top, right, bottom, left) tuple
        min_size_ratio: Minimum face size as fraction of frame

    Returns:
        Tuple of (is_valid, message)
    """
    if min_size_ratio is None:
        min_size_ratio = config.MIN_FACE_SIZE

    top, right, bottom, left = face_location
    face_height = bottom - top
    face_width = right - left
    face_area = face_height * face_width
    frame_area = image.shape[0] * image.shape[1]

    size_ratio = face_area / frame_area

    if size_ratio < min_size_ratio:
        return False, "Face is too small. Please move closer to the camera."

    # Extract face region and check brightness
    face = image[top:bottom, left:right]
    if len(face.shape) == 3:
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    else:
        gray = face

    mean_brightness = np.mean(gray)

    if mean_brightness < 50:
        return False, "Face is too dark. Please improve lighting."

    if mean_brightness > 220:
        return False, "Face is overexposed. Please reduce lighting."

    return True, "Face quality is good."


# =============================================================================
# FACE BOUNDING BOX UTILITIES
# =============================================================================

def face_location_to_bbox(
    face_location: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Convert face_recognition location to standard bbox format.

    face_recognition uses: (top, right, bottom, left)
    Standard bbox uses: (x1, y1, x2, y2) = (left, top, right, bottom)

    Args:
        face_location: (top, right, bottom, left) tuple

    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    top, right, bottom, left = face_location
    return (left, top, right, bottom)


def bbox_to_face_location(
    bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Convert standard bbox to face_recognition location format.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        Tuple of (top, right, bottom, left)
    """
    x1, y1, x2, y2 = bbox
    return (y1, x2, y2, x1)


def draw_face_rectangle(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw a rectangle around a detected face on the frame.

    Args:
        frame: Input frame (BGR format for OpenCV display)
        face_location: (top, right, bottom, left) tuple
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels
        label: Optional text label to display above rectangle

    Returns:
        np.ndarray: Frame with rectangle drawn

    Example:
        >>> faces = detect_faces_in_frame(frame)
        >>> for face in faces:
        ...     frame = draw_face_rectangle(frame, face, label="Student")
    """
    # Make a copy to avoid modifying original
    output = frame.copy()

    # Convert to standard bbox (left, top, right, bottom)
    left, top, right, bottom = face_location_to_bbox(face_location)

    # Draw rectangle
    cv2.rectangle(output, (left, top), (right, bottom), color, thickness)

    # Add label if provided
    if label:
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Draw background rectangle for text
        cv2.rectangle(
            output,
            (left, top - text_size[1] - 10),
            (left + text_size[0], top),
            color,
            -1  # Filled rectangle
        )

        # Add text
        cv2.putText(
            output,
            label,
            (left, top - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )

    return output


def draw_all_faces(
    frame: np.ndarray,
    face_locations: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw rectangles around all detected faces.

    Args:
        frame: Input frame (BGR format)
        face_locations: List of face location tuples
        labels: Optional list of labels (one per face)
        color: BGR color for rectangles

    Returns:
        np.ndarray: Frame with all faces marked
    """
    output = frame.copy()

    if labels is None:
        labels = [None] * len(face_locations)

    for face_loc, label in zip(face_locations, labels):
        output = draw_face_rectangle(
            output,
            face_loc,
            color=color,
            label=label
        )

    return output


# =============================================================================
# FACE DETECTION FOR REGISTRATION
# =============================================================================

def capture_good_face_for_registration(
    frame: np.ndarray
) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]]]:
    """
    Validate a frame for student registration.

    Checks:
    - Exactly one face present
    - Face is good quality
    - Face is well-positioned

    Args:
        frame: Input frame from webcam

    Returns:
        Tuple of (is_valid, message, face_location_or_None)
    """
    # First, detect faces
    valid, message, face_loc = validate_single_face(
        convert_bgr_to_rgb(frame)
    )

    if not valid:
        return False, message, None

    # Then validate quality
    rgb_frame = convert_bgr_to_rgb(frame)
    valid, message = validate_face_quality(rgb_frame, face_loc)

    if not valid:
        return False, message, None

    return True, "Ready for registration!", face_loc
