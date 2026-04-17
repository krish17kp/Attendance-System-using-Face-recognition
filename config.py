"""Central configuration for the Face Attendance System."""

from pathlib import Path

# App
APP_TITLE = "Face Attendance System"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EXPORTS_DIR = DATA_DIR / "exports"
DATABASE_PATH = DATA_DIR / "attendance.db"

for _d in [DATA_DIR, FACES_DIR, EXPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
FRAME_RESIZE_FACTOR = 0.5

# Face recognition
RECOGNITION_THRESHOLD = 0.50      # Euclidean distance cutoff
NUM_REGISTRATION_IMAGES = 15      # Target photos per student
NUM_ENCODINGS_PER_STUDENT = 10    # How many encodings to keep
MIN_FACE_SIZE = 0.03              # Min face area / frame area
MIN_QUALITY_SCORE = 0.25

# Attendance
ATTENDANCE_COOLDOWN = 5           # Seconds between re-mark attempts

# Formats
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

# Subjects / sections
DEFAULT_SUBJECTS = [
    "Mathematics", "Physics", "Chemistry", "Computer Science",
    "English", "Data Science", "Machine Learning", "Other",
]
DEPARTMENTS = ["Data Science", "CSE", "EEE", "MECH", "CIVIL", "IT", "AIML", "Other"]
YEARS = [1, 2, 3, 4]
SECTIONS = ["A", "B", "C", "D", "E"]