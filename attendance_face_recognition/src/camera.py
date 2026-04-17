"""
Camera module for the Attendance Face Recognition System.

Provides a clean wrapper around OpenCV's VideoCapture for:
- Camera initialization with error handling
- Frame capture with retry logic
- Camera settings configuration
- Proper resource cleanup
- Single frame and continuous capture modes

This module abstracts away the low-level OpenCV camera operations
for use in registration, attendance marking, and liveness detection.
"""

from typing import Optional, Tuple, List, Generator, Dict, Any
import cv2
import numpy as np

import config


# =============================================================================
# CAMERA WRAPPER CLASS
# =============================================================================

class Camera:
    """
    Wrapper class for webcam capture.

    Provides:
    - Safe camera initialization
    - Frame capture with quality checks
    - Automatic resource cleanup
    - Camera settings control

    Example:
        >>> camera = Camera()
        >>> camera.start()
        >>> ret, frame = camera.read()
        >>> camera.release()
    """

    def __init__(self, camera_index: int = None):
        """
        Initialize camera wrapper.

        Args:
            camera_index: Camera device index (0 = default webcam)
        """
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def start(self) -> Tuple[bool, str]:
        """
        Open and initialize the camera.

        Returns:
            Tuple of (success, message)
        """
        try:
            self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                return False, f"Could not open camera {self.camera_index}"

            # Configure camera settings
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            # Verify settings (some cameras don't support all settings)
            actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            self._is_opened = True

            return True, f"Camera opened successfully ({int(actual_width)}x{int(actual_height)})"

        except Exception as e:
            return False, f"Camera initialization failed: {str(e)}"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Returns:
            Tuple of (success, frame or None)
        """
        if not self._is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if ret and frame is not None:
            return True, frame
        else:
            return False, None

    def read_retry(self, max_retries: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a frame with retry logic.

        Useful for handling occasional dropped frames.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, frame or None)
        """
        for attempt in range(max_retries):
            ret, frame = self.read()

            if ret and frame is not None:
                return True, frame

        return False, None

    def release(self):
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        """Check if camera is opened and ready."""
        return self._is_opened

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.

        Returns:
            Tuple of (width, height)
        """
        if not self._is_opened or self._cap is None:
            return 0, 0

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def set_exposure(self, value: float) -> bool:
        """
        Set camera exposure value.

        Args:
            value: Exposure value (camera-dependent range)

        Returns:
            bool: Success status
        """
        if not self._is_opened or self._cap is None:
            return False

        # Note: Exposure control is camera-dependent
        # May not work on all cameras
        try:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, value)
            return True
        except:
            return False

    def set_brightness(self, value: float) -> bool:
        """
        Set camera brightness.

        Args:
            value: Brightness value (0-1 or camera-dependent)

        Returns:
            bool: Success status
        """
        if not self._is_opened or self._cap is None:
            return False

        try:
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
            return True
        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures camera is released."""
        self.release()


# =============================================================================
# CONTINUOUS CAPTURE
# =============================================================================

def continuous_capture(
    camera_index: int = 0,
    max_frames: int = 100,
    delay_ms: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Generator for continuous frame capture.

    Yields frames until max_frames reached or camera fails.

    Args:
        camera_index: Camera device index
        max_frames: Maximum frames to yield
        delay_ms: Delay between frames (milliseconds)

    Yields:
        np.ndarray: Captured frames

    Example:
        >>> for frame in continuous_capture(max_frames=60):
        ...     process_frame(frame)
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return

    try:
        for _ in range(max_frames):
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            yield frame

            # Small delay to control frame rate
            if cv2.waitKey(delay_ms) & 0xFF == 27:  # ESC to stop
                break

    finally:
        cap.release()


def capture_frames_batch(
    camera_index: int = 0,
    num_frames: int = 30,
    capture_delay: float = 0.5
) -> List[np.ndarray]:
    """
    Capture a batch of frames for registration or liveness check.

    Args:
        camera_index: Camera device index
        num_frames: Number of frames to capture
        capture_delay: Delay between captures (seconds)

    Returns:
        List of captured frames
    """
    import time

    frames = []
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return frames

    try:
        print(f"Capturing {num_frames} frames...")

        for i in range(num_frames):
            ret, frame = cap.read()

            if ret and frame is not None:
                frames.append(frame)
                print(f"  Captured {i + 1}/{num_frames}")

            time.sleep(capture_delay)

    finally:
        cap.release()

    return frames


# =============================================================================
# SINGLE FRAME CAPTURE
# =============================================================================

def capture_single_frame(
    camera_index: int = 0,
    show_preview: bool = False,
    window_name: str = "Camera Preview"
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Capture a single frame from camera.

    Optionally shows preview window.

    Args:
        camera_index: Camera device index
        show_preview: Show preview window
        window_name: Name for preview window

    Returns:
        Tuple of (success, frame or None)
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return False, None

    try:
        # Warm up the camera (first few frames may be dark)
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()

        if ret and frame is not None and show_preview:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()

        return ret, frame

    finally:
        cap.release()


def capture_frame_with_delay(
    camera_index: int = 0,
    delay_seconds: float = 3.0,
    countdown: bool = True
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Capture a frame after a countdown delay.

    Useful for giving user time to position themselves.

    Args:
        camera_index: Camera device index
        delay_seconds: Delay before capture
        countdown: Show countdown in preview

    Returns:
        Tuple of (success, frame or None)
    """
    import time

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return False, None

    try:
        # Warm up camera
        for _ in range(5):
            cap.read()

        # Countdown with preview
        if countdown:
            for i in range(int(delay_seconds), 0, -1):
                ret, frame = cap.read()

                if ret and frame is not None:
                    # Add countdown text
                    cv2.putText(
                        frame,
                        f"Capture in {i}...",
                        (50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
                    cv2.imshow("Preparing Capture", frame)
                    cv2.waitKey(1000)

            cv2.destroyAllWindows()

        else:
            time.sleep(delay_seconds)

        # Capture frame
        ret, frame = cap.read()

        return ret, frame

    finally:
        cap.release()


# =============================================================================
# CAMERA UTILITY FUNCTIONS
# =============================================================================

def get_available_cameras() -> List[int]:
    """
    Detect available camera devices.

    Returns:
        List of camera indices that are available
    """
    available = []

    # Check first 10 indices (usually 0-1 is enough for laptops)
    for i in range(10):
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            available.append(i)
            cap.release()

    return available


def test_camera(camera_index: int = 0) -> Dict[str, Any]:
    """
    Test camera functionality and get details.

    Args:
        camera_index: Camera device index

    Returns:
        Dict with camera information
    """
    result = {
        'available': False,
        'index': camera_index,
        'width': 0,
        'height': 0,
        'fps': 0,
        'error': None
    }

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        result['error'] = "Could not open camera"
        return result

    try:
        result['available'] = True
        result['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result['fps'] = int(cap.get(cv2.CAP_PROP_FPS))

        # Try to capture a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            result['error'] = "Could not capture frame"

    except Exception as e:
        result['error'] = str(e)

    finally:
        cap.release()

    return result


def check_camera_permission() -> Tuple[bool, str]:
    """
    Check if the application has permission to access the camera.

    Returns:
        Tuple of (has_permission, message)
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        return False, "Cannot access camera. Please check permissions and ensure no other application is using the camera."

    cap.release()
    return True, "Camera access OK"


# =============================================================================
# STREAMLIT COMPATIBLE CAMERA
# =============================================================================

class StreamlitCamera:
    """
    Camera wrapper optimized for Streamlit applications.

    Provides methods compatible with Streamlit's st.video and st.image.
    """

    def __init__(self, camera_index: int = 0):
        """Initialize Streamlit-compatible camera."""
        self.camera = Camera(camera_index)

    def get_frame_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture frame in RGB format (for Streamlit display).

        Returns:
            Tuple of (success, RGB frame or None)
        """
        success, frame = self.camera.read()

        if success and frame is not None:
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, rgb_frame

        return False, None

    def get_frame_as_image(self) -> Optional[np.ndarray]:
        """
        Get frame ready for st.image().

        Returns:
            RGB frame or None
        """
        _, frame = self.get_frame_rgb()
        return frame

    def start(self) -> bool:
        """Start camera."""
        success, _ = self.camera.start()
        return success

    def stop(self):
        """Stop camera."""
        self.camera.release()
