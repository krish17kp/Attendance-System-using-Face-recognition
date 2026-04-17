# Attendance System Using Face Recognition

A Streamlit-based face recognition attendance system for registering students, capturing face encodings, marking attendance in real time, and generating reports and analytics from stored attendance data.

## Project Overview

This project is a classroom attendance management application that uses facial recognition to automate student attendance. It combines:

- Streamlit for the web interface
- OpenCV for camera access and image processing
- `face_recognition` and `dlib` for face detection and face embeddings
- MediaPipe for liveness checks
- SQLite for local data storage
- Pandas and OpenPyXL for reporting and export

The application is organized as a multi-page Streamlit app with dedicated pages for registration, attendance, reports, and analytics.

## Key Features

- Student registration with form-based details entry
- Automatic multi-image face capture during registration
- Face encoding generation and storage
- Real-time face recognition for attendance marking
- Session-based attendance workflow by subject and section
- Duplicate attendance prevention per student, subject, and date
- Basic liveness detection to reduce spoofing with static photos
- Attendance filtering, CSV export, and Excel export
- Analytics for trends, subjects, sections, and student performance

## Pages

### Home

The home page provides:

- quick navigation to the major app modules
- total registered students
- attendance session status
- current recognition threshold

### Register Student

This page lets the user:

- enter student ID, name, department, year, and section
- capture multiple face images from the webcam
- generate and store face encodings
- save cropped face images for local reference

### Mark Attendance

This page lets the user:

- start an attendance session for a subject and section
- detect and recognize faces in the live camera feed
- optionally run liveness checks
- mark attendance automatically

### View Reports

This page supports:

- date-range filtering
- filtering by student, subject, section, and status
- detailed attendance tables
- CSV and Excel export
- summary generation

### Analytics

This page provides:

- attendance overview metrics
- recent attendance trends
- department and section analysis
- subject-wise comparisons
- top performers and low-attendance students

## Project Structure

```text
attendance_face_recognition/
├── app.py
├── config.py
├── requirements.txt
├── packages.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── 1_Register_Student.py
│   ├── 2_Mark_Attendance.py
│   ├── 3_View_Reports.py
│   └── 4_Analytics.py
├── src/
│   ├── attendance_manager.py
│   ├── camera.py
│   ├── database.py
│   ├── face_detector.py
│   ├── face_encoder.py
│   ├── liveness.py
│   ├── preprocess.py
│   ├── recognizer.py
│   ├── registration.py
│   ├── report_generator.py
│   └── utils.py
├── assets/
├── data/
└── models/
```

## How the System Works

### 1. Registration Workflow

During registration, the app:

1. validates student information
2. checks whether the student ID already exists
3. opens the webcam
4. captures multiple face frames when the face is well positioned
5. generates face encodings from the captured frames
6. stores the student record in SQLite
7. stores multiple face encodings for the same student for better recognition stability

### 2. Recognition Workflow

During attendance, the app:

1. loads all stored encodings from the database
2. detects faces in the live video feed
3. generates an embedding for each detected face
4. compares the embedding with stored encodings using Euclidean distance
5. uses the configured threshold to decide whether the face matches a known student
6. applies session and duplicate checks before writing attendance

### 3. Attendance Storage

Attendance is stored in SQLite with:

- student ID
- student name
- subject
- section
- date
- time
- status

The database enforces one attendance record per student, subject, and date.

## Technology Stack

- Python 3.11+
- Streamlit
- OpenCV
- face_recognition
- dlib
- MediaPipe
- SQLite
- Pandas
- OpenPyXL

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/krish17kp/Attendance-System-using-Face-recognition.git
cd Attendance-System-using-Face-recognition
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the App Locally

Use the project virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

The Streamlit app entry point is:

```text
app.py
```

## Configuration

The main configuration lives in `config.py`.

Important settings include:

- camera index
- frame size
- recognition threshold
- number of registration images
- attendance cooldown
- default subjects

## Data and Privacy Notes

This repository should not upload real face images, local databases, or generated encodings.

The following are intentionally excluded from version control:

- `data/attendance.db`
- `data/raw_faces/`
- `data/processed_faces/`
- `data/exports/`
- `models/encodings.pkl`
- `.venv/`

If you want sample data in the future, create sanitized demo data rather than uploading real student face data.

## Streamlit Deployment

This repository is prepared for Streamlit deployment with:

- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`

### Deploy on Streamlit Community Cloud

1. Push this project to GitHub
2. Open Streamlit Community Cloud
3. Choose **New app**
4. Select the repository:
   `krish17kp/Attendance-System-using-Face-recognition`
5. Select the branch:
   `branch1`
6. Set the main file path to:
   `app.py`
7. Deploy

### Important Deployment Note

This project depends on OpenCV, MediaPipe, `face_recognition`, and `dlib`. These libraries can be heavier than a standard Streamlit app and may require additional build support depending on the deployment environment.

If deployment fails on Streamlit Cloud, the likely reason is native dependency support rather than a problem in the Streamlit code itself.

## Current Limitations

- Uses local SQLite storage instead of a production database
- Recognition accuracy depends on lighting and camera quality
- Public cloud deployment may be constrained by native dependencies
- No authentication or role-based access control
- Attendance data is currently local to the running instance

## Suggested Future Improvements

- Admin login and access control
- Better audit logging
- Cloud database support
- Batch registration
- Face anti-spoofing improvements
- Attendance correction workflow
- Improved deployment pipeline

## Repository Name

Recommended display name:

**Attendance System Using Face Recognition**

Recommended short description:

**A Streamlit-based face recognition attendance system with student registration, real-time attendance marking, reporting, and analytics.**

## Author

Created and maintained by `krish17kp`.
