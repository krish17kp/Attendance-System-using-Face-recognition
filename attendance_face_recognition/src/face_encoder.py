"""
Face encoding module for the Attendance Face Recognition System.

Generates 128-dimensional face embeddings using the face_recognition library
(which uses dlib's ResNet-based model internally).

Key concepts:
- Face encoding: A 128D vector that uniquely represents a face
- Same person's faces have similar encodings (small Euclidean distance)
- Different people's faces have dissimilar encodings (large Euclidean distance)
- The encoding is invariant to lighting, pose, and expression variations

This module handles:
- Generating encodings from face images
- Batch encoding for multiple faces
- Encoding quality validation
- Managing encodings for storage and retrieval
"""

from typing import List, Tuple, Optional, Dict, Any
import warnings
import numpy as np
import cv2

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import face_recognition

import config
from .preprocess import crop_face_from_image, convert_bgr_to_rgb


# =============================================================================
# CORE FACE ENCODING
# =============================================================================

def generate_face_encoding(
    image: np.ndarray,
    face_location: Optional[Tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    """
    Generate a 128D face encoding for a detected face.

    Args:
        image: Input image (RGB format recommended)
        face_location: Optional (top, right, bottom, left) tuple
                      If None, detects faces automatically

    Returns:
        np.ndarray: 128D face encoding vector, or None if encoding failed

    Example:
        >>> image = cv2.imread('face.jpg')
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> encoding = generate_face_encoding(image_rgb)
        >>> print(f"Encoding shape: {encoding.shape}")
        (128,)
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if BGR (from OpenCV) by looking at typical values
        # Safe assumption: if we're not sure, convert
        rgb_image = convert_bgr_to_rgb(image) if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]) else image
    else:
        rgb_image = image

    # If face location not provided, detect faces
    if face_location is None:
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1)

        if len(face_locations) == 0:
            return None
        elif len(face_locations) > 1:
            # Use the largest (closest) face
            face_location = max(
                face_locations,
                key=lambda f: (f[2] - f[0]) * (f[1] - f[3])
            )
        else:
            face_location = face_locations[0]

    # Generate encoding
    # face_recognition.face_encodings returns a list of encodings
    encodings = face_recognition.face_encodings(rgb_image, [face_location])

    if len(encodings) > 0:
        return encodings[0]

    return None


def generate_face_encoding_from_frame(
    frame: np.ndarray,
    face_location: Optional[Tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    """
    Generate face encoding from a webcam frame.

    Handles BGR to RGB conversion automatically.

    Args:
        frame: Input frame from webcam (BGR format)
        face_location: Optional face location tuple

    Returns:
        np.ndarray: 128D face encoding, or None if failed
    """
    # Convert BGR to RGB
    rgb_frame = convert_bgr_to_rgb(frame)

    return generate_face_encoding(rgb_frame, face_location)


# =============================================================================
# BATCH ENCODING
# =============================================================================

def generate_encodings_for_multiple_faces(
    image: np.ndarray
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Generate encodings for all faces in an image.

    Args:
        image: Input image (RGB format)

    Returns:
        List of tuples (encoding, face_location)
        Empty list if no faces detected
    """
    # Detect all faces
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)

    if len(face_locations) == 0:
        return []

    # Generate encodings for all faces at once (more efficient)
    encodings = face_recognition.face_encodings(image, face_locations)

    # Pair encodings with locations
    return list(zip(encodings, face_locations))


def generate_multiple_encodings_for_registration(
    frames: List[np.ndarray],
    student_id: str,
    min_quality_score: float = 0.3
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Generate multiple face encodings from a sequence of frames.

    Used during student registration to capture multiple encodings
    for better recognition robustness.

    Args:
        frames: List of frames captured during registration
        student_id: Student's unique identifier
        min_quality_score: Minimum quality threshold for encodings

    Returns:
        Tuple of (list of good encodings, metadata dict)
    """
    encodings = []
    metadata = {
        'student_id': student_id,
        'total_frames': len(frames),
        'successful_encodings': 0,
        'failed_encodings': 0,
        'quality_scores': []
    }

    for i, frame in enumerate(frames):
        try:
            # Convert to RGB
            rgb_frame = convert_bgr_to_rgb(frame)

            # Detect face
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)

            if len(face_locations) == 0:
                metadata['failed_encodings'] += 1
                continue

            # Use largest face if multiple detected
            face_location = max(
                face_locations,
                key=lambda f: (f[2] - f[0]) * (f[1] - f[3])
            )

            # Generate encoding
            encoding = face_recognition.face_encodings(rgb_frame, [face_location])

            if len(encoding) > 0:
                encodings.append(encoding[0])

                # Calculate a simple quality score based on face size
                face_area = (face_location[2] - face_location[0]) * (face_location[1] - face_location[3])
                frame_area = frame.shape[0] * frame.shape[1]
                quality_score = face_area / frame_area

                metadata['quality_scores'].append(quality_score)

                if quality_score >= min_quality_score:
                    metadata['successful_encodings'] += 1
                else:
                    metadata['failed_encodings'] += 1
            else:
                metadata['failed_encodings'] += 1

        except Exception as e:
            print(f"Error encoding frame {i} for student {student_id}: {e}")
            metadata['failed_encodings'] += 1

    return encodings, metadata


# =============================================================================
# ENCODING VALIDATION
# =============================================================================

def validate_encoding(
    encoding: np.ndarray
) -> Tuple[bool, str]:
    """
    Validate a face encoding vector.

    Checks:
    - Correct shape (128,)
    - No NaN or Inf values
    - Reasonable magnitude

    Args:
        encoding: 128D numpy array

    Returns:
        Tuple of (is_valid, message)
    """
    # Check shape
    if encoding is None:
        return False, "Encoding is None"

    if not isinstance(encoding, np.ndarray):
        return False, "Encoding is not a numpy array"

    if encoding.shape != (128,):
        return False, f"Invalid encoding shape: {encoding.shape}, expected (128,)"

    # Check for NaN or Inf
    if np.any(np.isnan(encoding)):
        return False, "Encoding contains NaN values"

    if np.any(np.isinf(encoding)):
        return False, "Encoding contains Inf values"

    # Check magnitude (typical face encodings have norm around 1.0)
    magnitude = np.linalg.norm(encoding)
    if magnitude < 0.1 or magnitude > 10.0:
        return False, f"Unusual encoding magnitude: {magnitude:.4f}"

    return True, "Encoding is valid"


def validate_encodings_batch(
    encodings: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Validate a batch of encodings and return only valid ones.

    Args:
        encodings: List of encoding arrays

    Returns:
        Tuple of (valid_encodings, indices_of_valid_encodings)
    """
    valid_encodings = []
    valid_indices = []

    for i, encoding in enumerate(encodings):
        is_valid, _ = validate_encoding(encoding)
        if is_valid:
            valid_encodings.append(encoding)
            valid_indices.append(i)

    return valid_encodings, valid_indices


# =============================================================================
# ENCODING STATISTICS
# =============================================================================

def get_encoding_statistics(
    encodings: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calculate statistics for a set of encodings.

    Useful for debugging and quality assessment.

    Args:
        encodings: List of encoding arrays

    Returns:
        Dict with mean, std, min, max magnitude statistics
    """
    if not encodings:
        return {
            'count': 0,
            'mean_magnitude': 0.0,
            'std_magnitude': 0.0,
            'min_magnitude': 0.0,
            'max_magnitude': 0.0
        }

    magnitudes = [np.linalg.norm(enc) for enc in encodings]

    return {
        'count': len(encodings),
        'mean_magnitude': float(np.mean(magnitudes)),
        'std_magnitude': float(np.std(magnitudes)),
        'min_magnitude': float(np.min(magnitudes)),
        'max_magnitude': float(np.max(magnitudes))
    }


def calculate_encoding_variance(
    encodings: List[np.ndarray]
) -> float:
    """
    Calculate variance among multiple encodings of the same person.

    Lower variance = more consistent encodings = better recognition.

    Args:
        encodings: List of encoding arrays from same person

    Returns:
        float: Average pairwise distance (lower is better)
    """
    if len(encodings) < 2:
        return 0.0

    from .utils import euclidean_distance

    distances = []
    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            dist = euclidean_distance(encodings[i], encodings[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


# =============================================================================
# ENCODING MANAGEMENT
# =============================================================================

def select_best_encodings(
    encodings: List[np.ndarray],
    quality_scores: List[float],
    num_to_select: int = 5
) -> List[np.ndarray]:
    """
    Select the best encodings based on quality scores.

    During registration, we may capture many encodings.
    This function selects the best ones for storage.

    Args:
        encodings: List of encoding arrays
        quality_scores: List of quality scores (same length as encodings)
        num_to_select: Number of encodings to select

    Returns:
        List of best encodings
    """
    if len(encodings) == 0:
        return []

    # Sort by quality score (descending)
    sorted_indices = np.argsort(quality_scores)[::-1]

    # Select top N
    selected_indices = sorted_indices[:min(num_to_select, len(encodings))]

    return [encodings[i] for i in selected_indices]


def merge_encodings(
    existing_encodings: List[np.ndarray],
    new_encodings: List[np.ndarray],
    max_encodings: int = 10
) -> List[np.ndarray]:
    """
    Merge new encodings with existing ones, keeping the best.

    Useful when adding more face samples for an existing student.

    Args:
        existing_encodings: Currently stored encodings
        new_encodings: Newly captured encodings
        max_encodings: Maximum encodings to keep

    Returns:
        Merged list of best encodings
    """
    all_encodings = existing_encodings + new_encodings

    if len(all_encodings) <= max_encodings:
        return all_encodings

    # Calculate quality based on distance to mean
    mean_encoding = np.mean(all_encodings, axis=0)

    quality_scores = [
        1.0 / (1.0 + np.linalg.norm(enc - mean_encoding))
        for enc in all_encodings
    ]

    return select_best_encodings(all_encodings, quality_scores, max_encodings)


# =============================================================================
# ENCODING VISUALIZATION (DEBUG)
# =============================================================================

def visualize_encoding_distribution(
    encodings: List[np.ndarray]
) -> str:
    """
    Generate a text-based visualization of encoding distribution.

    Useful for debugging without requiring plotting libraries.

    Args:
        encodings: List of encoding arrays

    Returns:
        str: Text representation of encoding statistics
    """
    if not encodings:
        return "No encodings to visualize"

    stats = get_encoding_statistics(encodings)

    output = []
    output.append("=" * 50)
    output.append("FACE ENCODING STATISTICS")
    output.append("=" * 50)
    output.append(f"Count: {stats['count']}")
    output.append(f"Mean magnitude: {stats['mean_magnitude']:.4f}")
    output.append(f"Std magnitude: {stats['std_magnitude']:.4f}")
    output.append(f"Min magnitude: {stats['min_magnitude']:.4f}")
    output.append(f"Max magnitude: {stats['max_magnitude']:.4f}")

    if len(encodings) > 1:
        variance = calculate_encoding_variance(encodings)
        output.append(f"Average pairwise distance: {variance:.4f}")

    output.append("=" * 50)

    return "\n".join(output)
