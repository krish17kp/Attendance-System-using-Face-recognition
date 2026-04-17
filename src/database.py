"""
database.py — all SQLite operations for the Face Attendance System.

Tables:
  students   – registration info
  encodings  – 128-D face vectors (BLOB, one row per capture)
  attendance – daily attendance records
"""

import pickle
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _conn(read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{config.DATABASE_PATH.resolve().as_posix()}?mode=ro"
        c = sqlite3.connect(uri, uri=True)
    else:
        c = sqlite3.connect(config.DATABASE_PATH)
        c.execute("PRAGMA foreign_keys = ON")
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables if they don't exist."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT    UNIQUE NOT NULL,
                name        TEXT    NOT NULL,
                department  TEXT    NOT NULL,
                year        INTEGER NOT NULL,
                section     TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS encodings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT    NOT NULL,
                encoding    BLOB    NOT NULL,
                image_path  TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT    NOT NULL,
                name        TEXT    NOT NULL,
                subject     TEXT    NOT NULL,
                section     TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                time        TEXT    NOT NULL,
                status      TEXT    DEFAULT 'Present',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                UNIQUE(student_id, subject, date)
            );
        """)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def insert_student(student_id: str, name: str, department: str,
                   year: int, section: str) -> bool:
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO students (student_id,name,department,year,section) VALUES (?,?,?,?,?)",
                (student_id, name, department, year, section),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_student(student_id: str) -> Optional[Dict[str, Any]]:
    with _conn(read_only=True) as c:
        row = c.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
    return dict(row) if row else None


def get_all_students() -> List[Dict[str, Any]]:
    with _conn(read_only=True) as c:
        rows = c.execute("SELECT * FROM students ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def delete_student(student_id: str) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        c.execute("DELETE FROM encodings WHERE student_id=?", (student_id,))
        c.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    return True


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def insert_encoding(student_id: str, encoding: bytes,
                    image_path: Optional[str] = None) -> bool:
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO encodings (student_id,encoding,image_path) VALUES (?,?,?)",
                (student_id, encoding, image_path),
            )
        return True
    except sqlite3.Error:
        return False


def get_all_encodings() -> List[Dict[str, Any]]:
    """Return every stored encoding with student name (for recognition)."""
    with _conn(read_only=True) as c:
        rows = c.execute("""
            SELECT e.student_id, s.name, e.encoding
            FROM encodings e
            JOIN students s ON e.student_id = s.student_id
        """).fetchall()
    return [dict(r) for r in rows]


def get_student_encodings(student_id: str) -> List[bytes]:
    with _conn(read_only=True) as c:
        rows = c.execute(
            "SELECT encoding FROM encodings WHERE student_id=?", (student_id,)
        ).fetchall()
    return [r["encoding"] for r in rows]


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

def attendance_exists(student_id: str, subject: str, date: str) -> bool:
    with _conn(read_only=True) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM attendance WHERE student_id=? AND subject=? AND date=?",
            (student_id, subject, date),
        ).fetchone()[0]
    return n > 0


def mark_attendance(student_id: str, name: str, subject: str,
                    section: str, date: str, time: str,
                    status: str = "Present") -> Tuple[bool, str]:
    if attendance_exists(student_id, subject, date):
        return False, "Already marked"
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO attendance (student_id,name,subject,section,date,time,status)"
                " VALUES (?,?,?,?,?,?,?)",
                (student_id, name, subject, section, date, time, status),
            )
        return True, "Marked successfully"
    except sqlite3.IntegrityError:
        return False, "Duplicate entry"


def get_attendance_records(
    student_id: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query, params = "SELECT * FROM attendance WHERE 1=1", []
    if student_id:
        query += " AND student_id=?";  params.append(student_id)
    if subject:
        query += " AND subject=?";     params.append(subject)
    if date_from:
        query += " AND date>=?";       params.append(date_from)
    if date_to:
        query += " AND date<=?";       params.append(date_to)
    if section:
        query += " AND section=?";     params.append(section)
    query += " ORDER BY date DESC, time DESC"
    with _conn(read_only=True) as c:
        rows = c.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_student_attendance_summary(student_id: str) -> Dict[str, Any]:
    with _conn(read_only=True) as c:
        row = c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present
            FROM attendance WHERE student_id=?
        """, (student_id,)).fetchone()
    total   = row["total"]   or 0
    present = row["present"] or 0
    pct     = round(present / total * 100, 1) if total else 0.0
    return {"total_classes": total, "present_count": present, "percentage": pct}


# Run on import
create_tables()
