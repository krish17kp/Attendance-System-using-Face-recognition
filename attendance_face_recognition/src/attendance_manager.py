"""
attendance_manager.py — session management and attendance marking.

One session = one subject + section + date.
Prevents the same student marking twice per subject per day.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import config
from src.database import attendance_exists, get_all_students, mark_attendance
from src.utils import get_current_date, get_current_time


class AttendanceSession:
    """Tracks which students have been marked in the current session."""

    def __init__(self):
        self.subject: Optional[str] = None
        self.section: Optional[str] = None
        self.date:    str            = get_current_date()
        self.active:  bool           = False
        self._marked: set            = set()

    def start(self, subject: str, section: str, date: Optional[str] = None):
        self.subject = subject
        self.section = section
        self.date    = date or get_current_date()
        self.active  = True
        self._marked.clear()

    def end(self):
        self.active = False
        self._marked.clear()

    def already_marked(self, student_id: str) -> bool:
        return student_id in self._marked

    def add_marked(self, student_id: str):
        self._marked.add(student_id)

    @property
    def marked_count(self) -> int:
        return len(self._marked)


class AttendanceManager:
    """High-level controller: start/end sessions and mark attendance."""

    def __init__(self):
        self.session = AttendanceSession()
        self._cooldown: Dict[str, datetime] = {}
        self._daily_count: int = 0
        self._last_reset_date: str = get_current_date()

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start_session(self, subject: str, section: str,
                      date: Optional[str] = None) -> Tuple[bool, str]:
        if not subject.strip():
            return False, "Subject cannot be empty"
        self.session.start(subject, section, date)
        return True, f"Session started: {subject} – Section {section}"

    def end_session(self) -> Tuple[bool, str]:
        if not self.session.active:
            return False, "No active session"
        count = self.session.marked_count
        self.session.end()
        return True, f"Session ended. {count} students marked."

    # ------------------------------------------------------------------
    # Marking
    # ------------------------------------------------------------------

    def mark_attendance(
        self, student_id: str, name: str,
        subject: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Mark attendance for *student_id*.

        Checks (in order):
          1. Session active (or explicit subject provided)
          2. Not already marked for this subject/date
          3. Cooldown not exceeded
        """
        self._daily_reset_if_needed()

        subj = subject or (self.session.subject if self.session.active else None)
        sec  = section or (self.session.section if self.session.active else "N/A")

        if not subj:
            return False, "No active session", {}

        today = get_current_date()
        now   = datetime.now()

        # Already marked in DB?
        if attendance_exists(student_id, subj, today):
            return False, "Already marked today", {"duplicate": True}

        # Cooldown check
        last = self._cooldown.get(student_id)
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed < config.ATTENDANCE_COOLDOWN:
                wait = int(config.ATTENDANCE_COOLDOWN - elapsed)
                return False, f"Wait {wait}s", {}

        # Write to DB
        success, msg = mark_attendance(
            student_id, name, subj, sec, today, get_current_time()
        )
        if success:
            self._cooldown[student_id] = now
            self._daily_count += 1
            if self.session.active:
                self.session.add_marked(student_id)
            return True, "Attendance marked ✓", {
                "student_id": student_id, "name": name,
                "subject": subj, "date": today,
            }
        return False, msg, {}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_today_stats(self) -> Dict[str, Any]:
        self._daily_reset_if_needed()
        return {
            "date":             get_current_date(),
            "total_marked":     self._daily_count,
            "session_active":   self.session.active,
            "current_subject":  self.session.subject,
            "current_section":  self.session.section,
            "marked_in_session": self.session.marked_count,
        }

    def _daily_reset_if_needed(self):
        today = get_current_date()
        if today != self._last_reset_date:
            self._daily_count      = 0
            self._cooldown.clear()
            self._last_reset_date  = today