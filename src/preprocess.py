"""
Image preprocessing module for the Attendance Face Recognition System.

Handles all image preprocessing operations:
- Resizing frames for faster processing
- Color space conversion (BGR to RGB)
- Face region extraction/cropping
- Image quality filtering
- Normalization

Preprocessing is critical for:
- Faster face detection (smaller frames = faster processing)
- Consistent input to the face recognition model
- Better recognition accuracy
"""

from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import cv2

import config


# =============================================================================
# FRAME PREPROCESSING
# =============================================================================

def resize_frame(
    frame: np.ndarray,
    scale_factor: Optional[float] = None
) -> np.ndarray:
    """
    Resize frame for faster face detection.

    Processing smaller frames is significantly faster while still
    maintaining good face detection accuracy.

    Args:
        frame: Input frame (BGR format from OpenCV)
        scale_factor: Resize scale (uses config default if None)
                     1.0 = full size, 0.5 = half size, etc.

    Returns:
        np.ndarray: Resized frame

    Example:
        >>> frame = cv2.imread('image.jpg')
        >>> small_frame = resize_frame(frame, 0.5)
    """
    if scale_factor is None:
        scale_factor = config.FRAME_RESIZE_FACTOR

    if scale_factor >= 1.0:
        return frame

    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR frame to RGB format.

    OpenCV reads images in BGR format, but the face_recognition
    library (and most deep learning models) expect RGB format.

    Args:
        frame: Input frame in BGR format

    Returns:
        np.ndarray: Frame in RGB format

    Note:
        This is a simple channel swap, not a color space conversion.
        For grayscale conversion, use convert_to_grayscale().
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale.

    Some face detection algorithms work on grayscale images.
    Also useful for certain preprocessing pipelines.

    Args:
        frame: Input frame (BGR or RGB)

    Returns:
        np.ndarray: Grayscale frame (single channel)
    """
    if len(frame.shape) == 2:
        return frame  # Already grayscale

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def preprocess_frame(
    frame: np.ndarray,
    resize: bool = True,
    convert_color: bool = True,
    scale_factor: Optional[float] = None
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a frame.

    This is the main preprocessing function used before face detection.

    Pipeline:
    1. Resize frame (optional, for speed)
    2. Convert BGR to RGB (required for face_recognition library)

    Args:
        frame: Input frame from webcam (BGR format)
        resize: Whether to resize the frame
        convert_color: Whether to convert to RGB
        scale_factor: Resize scale factor

    Returns:
        np.ndarray: Preprocessed frame ready for face detection

    Example:
        >>> frame = camera.read()
        >>> processed = preprocess_frame(frame, resize=True)
        >>> face_locations = face_recognition.face_locations(processed)
    """
    result = frame.copy()

    # Step 1: Resize for faster processing
    if resize:
        result = resize_frame(result, scale_factor)

    # Step 2: Convert BGR to RGB for face_recognition library
    if convert_color:
        result = convert_bgr_to_rgb(result)

    return result


# =============================================================================
# FACE CROPPING AND EXTRACTION
# =============================================================================

def crop_face_from_image(
    image: np.ndarray,
    face_location: Tuple[int, int, int, int],
    padding: int = 10
) -> np.ndarray:
    """
    Extract face region from an image.

    Args:
        image: Input image containing face
        face_location: (top, right, bottom, left) tuple from face_recognition
        padding: Pixels of padding around face

    Returns:
        np.ndarray: Cropped face image

    Example:
        >>> face_locs = face_recognition.face_locations(image)
        >>> face_crop = crop_face_from_image(image, face_locs[0])
    """
    top, right, bottom, left = face_location

    # Add padding (ensure we don't go out of bounds)
    img_height, img_width = image.shape[:2]

    top = max(0, top - padding)
    right = min(img_width, right + padding)
    bottom = min(img_height, bottom + padding)
    left = max(0, left - padding)

    # Extract face region
    # Note: face_location uses (top, right, bottom, left) order
    # NumPy indexing uses [row, col] = [y, x] = [top:bottom, left:right]
    face = image[top:bottom, left:right]

    return face


def crop_face_from_frame(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    scale_factor: Optional[float] = None
) -> np.ndarray:
    """
    Extract face from frame, accounting for any resize that was applied.

    When you resize a frame before detection, the face locations are
    relative to the resized frame. This function handles the conversion.

    Args:
        frame: Original full-size frame (before resize)
        face_location: Face location from resized frame detection
        scale_factor: Scale factor used for resizing

    Returns:
        np.ndarray: Cropped face from original frame
    """
    if scale_factor is None:
        scale_factor = config.FRAME_RESIZE_FACTOR

    # Scale face location back to original frame size
    if scale_factor < 1.0:
        scale_inv = 1.0 / scale_factor
        top, right, bottom, left = face_location
        face_location = (
            int(top * scale_inv),
            int(right * scale_inv),
            int(bottom * scale_inv),
            int(left * scale_inv)
        )

    return crop_face_from_image(frame, face_location)


def expand_face_region(
    face_location: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    expansion_factor: float = 1.3
) -> Tuple[int, int, int, int]:
    """
    Expand face region to include more context (hair, neck, etc.).

    Useful for saving face images that look more natural.

    Args:
        face_location: (top, right, bottom, left) tuple
        image_shape: (height, width) of the image
        expansion_factor: How much to expand (1.0 = no expansion)

    Returns:
        Tuple[int, int, int, int]: Expanded face location
    """
    top, right, bottom, left = face_location
    height, width = image_shape

    # Calculate face dimensions
    face_height = bottom - top
    face_width = right - left

    # Calculate expansion amounts
    expand_height = int(face_height * (expansion_factor - 1) / 2)
    expand_width = int(face_width * (expansion_factor - 1) / 2)

    # Apply expansion with bounds checking
    new_top = max(0, top - expand_height)
    new_right = min(width, right + expand_width)
    new_bottom = min(height, bottom + expand_height)
    new_left = max(0, left - expand_width)

    return (new_top, new_right, new_bottom, new_left)


# =============================================================================
# IMAGE QUALITY FILTERING
# =============================================================================

def check_face_visibility(
    image: np.ndarray,
    face_location: Tuple[int, int, int, int]
) -> bool:
    """
    Check if the detected face region is clearly visible.

    Uses simple heuristics:
    - Not too dark
    - Not too bright
    - Has sufficient contrast

    Args:
        image: Full image
        face_location: (top, right, bottom, left) tuple

    Returns:
        bool: True if face appears visible
    """
    face = crop_face_from_image(image, face_location)
    gray = convert_to_grayscale(face)

    # Calculate mean brightness
    mean_brightness = np.mean(gray)

    # Calculate contrast (standard deviation)
    contrast = np.std(gray)

    # Check if face is well-lit and has contrast
    is_not_too_dark = mean_brightness > 50
    is_not_too_bright = mean_brightness < 220
    has_contrast = contrast > 20

    return is_not_too_dark and is_not_too_bright and has_contrast


def filter_low_quality_faces(
    face_images: List[np.ndarray],
    min_sharpness: float = 100.0
) -> List[Tuple[np.ndarray, bool]]:
    """
    Filter out low-quality face images.

    Args:
        face_images: List of cropped face images
        min_sharpness: Minimum sharpness threshold

    Returns:
        List of tuples (image, is_good_quality)
    """
    from .utils import calculate_image_sharpness

    results = []
    for face in face_images:
        gray = convert_to_grayscale(face)
        sharpness = calculate_image_sharpness(gray)
        is_good = sharpness >= min_sharpness
        results.append((face, is_good))

    return results


def select_best_face_image(
    face_images: List[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """
    Select the best quality face image from a list.

    Uses sharpness as the quality metric.

    Args:
        face_images: List of cropped face images

    Returns:
        Tuple of (best_image, index)
        Returns (first_image, 0) if list is empty
    """
    from .utils import calculate_image_sharpness

    if not face_images:
        return None, 0

    best_score = -1
    best_idx = 0

    for i, face in enumerate(face_images):
        gray = convert_to_grayscale(face)
        score = calculate_image_sharpness(gray)

        if score > best_score:
            best_score = score
            best_idx = i

    return face_images[best_idx], best_idx


# =============================================================================
# FACE NORMALIZATION
# =============================================================================

def normalize_face_image(
    face_image: np.ndarray,
    target_size: Tuple[int, int] = (150, 150)
) -> np.ndarray:
    """
    Normalize face image for consistent storage and display.

    Applies:
    1. Resize to standard size
    2. Histogram equalization (optional, for lighting normalization)

    Args:
        face_image: Cropped face image
        target_size: (width, height) for output

    Returns:
        np.ndarray: Normalized face image
    """
    # Resize to standard size
    normalized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_CUBIC)

    return normalized


def apply_histogram_equalization(
    image: np.ndarray,
    color: bool = False
) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.

    Args:
        image: Input image
        color: If True, apply to color image (converts to LAB space)
               If False, apply to grayscale

    Returns:
        np.ndarray: Image with enhanced contrast
    """
    if color:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # to the L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = convert_to_grayscale(image)
        else:
            gray = image

        # Apply global histogram equalization
        return cv2.equalizeHist(gray)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def preprocess_multiple_faces(
    images: List[np.ndarray],
    target_size: Tuple[int, int] = (150, 150)
) -> List[np.ndarray]:
    """
    Preprocess multiple face images consistently.

    Args:
        images: List of face images
        target_size: Target size for all images

    Returns:
        List of preprocessed images
    """
    return [
        normalize_face_image(img, target_size)
        for img in images
    ]


def save_face_image(
    face_image: np.ndarray,
    filepath: str,
    quality: int = 95
) -> bool:
    """
    Save a face image to disk.

    Args:
        face_image: Face image to save
        filepath: Full path to save the image
        quality: JPEG quality (1-100, higher = better quality)

    Returns:
        bool: True if save successful
    """
    try:
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save image
        cv2.imwrite(filepath, face_image, [
            int(cv2.IMWRITE_JPEG_QUALITY),
            quality
        ])

        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False
