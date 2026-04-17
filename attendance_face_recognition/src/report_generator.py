"""
Report generation module for the Attendance Face Recognition System.

Handles:
- Generating filtered attendance reports
- Exporting to CSV format
- Exporting to Excel format (.xlsx)
- Summary statistics
- Student-wise and subject-wise reports

Uses pandas for data manipulation and openpyxl for Excel export.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import csv

import pandas as pd

import config
from .database import (
    get_attendance_records,
    get_all_students,
    get_student,
    get_student_attendance_summary,
    get_subject_wise_summary
)
from .utils import get_current_date, format_date_display


# =============================================================================
# REPORT DATA GENERATION
# =============================================================================

class ReportGenerator:
    """
    Generate attendance reports with various filters and formats.

    Example:
        >>> generator = ReportGenerator()
        >>> df = generator.generate_report(
        ...     date_from="2026-04-01",
        ...     date_to="2026-04-12",
        ...     subject="Computer Vision"
        ... )
        >>> generator.export_csv(df, "report.csv")
    """

    def __init__(self):
        """Initialize the report generator."""
        self.exports_dir = config.EXPORTS_DIR
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        student_id: Optional[str] = None,
        subject: Optional[str] = None,
        section: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with attendance records.

        Args:
            student_id: Filter by student ID
            subject: Filter by subject
            section: Filter by section
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)

        Returns:
            pandas DataFrame with attendance records
        """
        # Get records from database
        records = get_attendance_records(
            student_id=student_id,
            subject=subject,
            date_from=date_from,
            date_to=date_to,
            section=section
        )

        # Convert to DataFrame
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Add formatted columns
        df['date_display'] = df['date'].apply(format_date_display)
        df['time_display'] = df['time'].apply(
            lambda t: datetime.strptime(t, config.TIME_FORMAT).strftime("%I:%M %p")
        )

        # Reorder columns for display
        display_columns = [
            'id', 'student_id', 'name', 'subject', 'section',
            'date', 'date_display', 'time', 'time_display', 'status'
        ]

        # Only include columns that exist
        available_columns = [c for c in display_columns if c in df.columns]
        df = df[available_columns]

        return df

    def generate_student_report(
        self,
        student_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate a comprehensive report for a single student.

        Args:
            student_id: Student's unique identifier

        Returns:
            Tuple of (DataFrame with records, summary dict)
        """
        # Get student info
        student = get_student(student_id)
        if not student:
            return pd.DataFrame(), {'error': 'Student not found'}

        # Get attendance records
        records = get_attendance_records(student_id=student_id)
        df = pd.DataFrame(records) if records else pd.DataFrame()

        # Get summary statistics
        summary = get_student_attendance_summary(student_id)
        summary['student_name'] = student.get('name', '')
        summary['department'] = student.get('department', '')
        summary['year'] = student.get('year', '')
        summary['section'] = student.get('section', '')

        return df, summary

    def generate_subject_report(
        self,
        subject: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate report for a specific subject.

        Args:
            subject: Subject name
            date_from: Start date
            date_to: End date

        Returns:
            Tuple of (DataFrame, summary statistics)
        """
        records = get_attendance_records(
            subject=subject,
            date_from=date_from,
            date_to=date_to
        )

        df = pd.DataFrame(records) if records else pd.DataFrame()

        # Calculate summary
        summary = {
            'subject': subject,
            'total_records': len(records),
            'unique_students': df['student_id'].nunique() if not df.empty else 0,
            'date_range': f"{date_from or 'All'} to {date_to or 'All'}"
        }

        return df, summary

    def generate_daily_report(
        self,
        date: str,
        subject: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate report for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            subject: Optional subject filter

        Returns:
            pandas DataFrame with daily attendance
        """
        return self.generate_report(
            date_from=date,
            date_to=date,
            subject=subject
        )

    def generate_summary_report(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a summary report with attendance statistics.

        Args:
            date_from: Start date
            date_to: End date

        Returns:
            pandas DataFrame with summary statistics
        """
        students = get_all_students()
        summary_data = []

        for student in students:
            student_id = student['student_id']
            summary = get_student_attendance_summary(
                student_id,
                subject=None  # Overall across all subjects
            )

            summary_data.append({
                'student_id': student_id,
                'name': student['name'],
                'department': student['department'],
                'year': student['year'],
                'section': student['section'],
                'total_classes': summary['total_classes'],
                'present_count': summary['present_count'],
                'attendance_percentage': summary['percentage']
            })

        return pd.DataFrame(summary_data)

    def export_to_csv(
        self,
        data,
        filepath: Optional[str] = None,
        include_index: bool = False
    ) -> Tuple[bool, str]:
        """
        Backward-compatible CSV export used by the Streamlit page.

        Accepts either a DataFrame or a list of record dicts.
        """
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return export_to_csv(df, filepath, include_index=include_index)

    def export_to_excel(
        self,
        data,
        filepath: Optional[str] = None,
        sheet_name: str = "Attendance"
    ) -> Tuple[bool, str]:
        """
        Backward-compatible Excel export used by the Streamlit page.

        Accepts either a DataFrame or a list of record dicts.
        """
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return export_to_excel(df, filepath, sheet_name=sheet_name)

    def generate_student_wise_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Generate the student-wise summary expected by the reports page.
        """
        students = get_all_students()
        report_data: List[Dict[str, Any]] = []

        for student in students:
            records = get_attendance_records(
                student_id=student['student_id'],
                date_from=start_date,
                date_to=end_date
            )

            total_classes = len(records)
            present_count = sum(1 for record in records if record.get('status') == 'Present')
            attendance_rate = round((present_count / total_classes) * 100, 2) if total_classes > 0 else 0.0

            report_data.append({
                'student_id': student['student_id'],
                'name': student['name'],
                'department': student['department'],
                'section': student['section'],
                'total_classes': total_classes,
                'present': present_count,
                'attendance_rate': attendance_rate
            })

        return True, report_data, "Student-wise report generated successfully"

    def generate_subject_wise_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Generate the subject-wise summary expected by the reports page.
        """
        records = get_attendance_records(
            date_from=start_date,
            date_to=end_date
        )

        if not records:
            return True, [], "No attendance records found for the selected date range"

        df = pd.DataFrame(records)
        report_data: List[Dict[str, Any]] = []

        for subject, group in df.groupby('subject'):
            total_records = len(group)
            present = int((group['status'] == 'Present').sum())
            absent = total_records - present
            total_students = group['student_id'].nunique()
            attendance_rate = round((present / total_records) * 100, 2) if total_records > 0 else 0.0

            report_data.append({
                'subject': subject,
                'total_students': int(total_students),
                'total_records': int(total_records),
                'present': present,
                'absent': absent,
                'attendance_rate': attendance_rate
            })

        report_data.sort(key=lambda item: item['attendance_rate'], reverse=True)
        return True, report_data, "Subject-wise report generated successfully"


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(
    df: pd.DataFrame,
    filepath: Optional[str] = None,
    include_index: bool = False
) -> Tuple[bool, str]:
    """
    Export DataFrame to CSV file.

    Args:
        df: pandas DataFrame to export
        filepath: Output file path (auto-generated if None)
        include_index: Whether to include index column

    Returns:
        Tuple of (success, message)
    """
    if df.empty:
        return False, "No data to export"

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = config.EXPORTS_DIR / f"attendance_export_{timestamp}.csv"
    else:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(filepath, index=include_index)
        return True, f"Exported to {filepath}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_to_excel(
    df: pd.DataFrame,
    filepath: Optional[str] = None,
    sheet_name: str = "Attendance"
) -> Tuple[bool, str]:
    """
    Export DataFrame to Excel file (.xlsx).

    Args:
        df: pandas DataFrame to export
        filepath: Output file path (auto-generated if None)
        sheet_name: Name of the sheet

    Returns:
        Tuple of (success, message)
    """
    if df.empty:
        return False, "No data to export"

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = config.EXPORTS_DIR / f"attendance_export_{timestamp}.xlsx"
    else:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        return True, f"Exported to {filepath}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


# =============================================================================
# EXPORT CLASS METHODS
# =============================================================================

    def export_report_csv(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Export report to CSV.

        Args:
            df: DataFrame to export
            filename: Optional filename (without path)

        Returns:
            Tuple of (success, message)
        """
        if filename:
            filepath = self.exports_dir / filename
        else:
            filepath = None

        return export_to_csv(df, filepath)

    def export_report_excel(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Export report to Excel.

        Args:
            df: DataFrame to export
            filename: Optional filename (without path)

        Returns:
            Tuple of (success, message)
        """
        if filename:
            filepath = self.exports_dir / filename
        else:
            filepath = None

        return export_to_excel(df, filepath)

    def export_student_report(
        self,
        student_id: str,
        format: str = 'excel'
    ) -> Tuple[bool, str, str]:
        """
        Export comprehensive student report.

        Args:
            student_id: Student's unique identifier
            format: 'csv' or 'excel'

        Returns:
            Tuple of (success, message, filepath)
        """
        df, summary = self.generate_student_report(student_id)

        if df.empty:
            return False, "No attendance records found", ""

        # Create summary DataFrame
        summary_df = pd.DataFrame([summary])

        # Combine summary and detailed records
        # Add empty rows for spacing
        spacer = pd.DataFrame([{}])
        combined = pd.concat([summary_df, spacer, df], ignore_index=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_{student_id}_report_{timestamp}"

        if format == 'excel':
            success, msg = self.export_report_excel(
                combined,
                f"{filename}.xlsx"
            )
        else:
            success, msg = self.export_report_csv(
                combined,
                f"{filename}.csv"
            )

        filepath = str(self.exports_dir / f"{filename}.{format}") if success else ""
        return success, msg, filepath


# =============================================================================
# STANDALONE EXPORT FUNCTIONS
# =============================================================================

def export_attendance_to_csv(
    student_id: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    filepath: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Quick export of attendance records to CSV.

    Args:
        student_id: Filter by student
        subject: Filter by subject
        date_from: Start date
        date_to: End date
        filepath: Output file path

    Returns:
        Tuple of (success, message)
    """
    generator = ReportGenerator()
    df = generator.generate_report(
        student_id=student_id,
        subject=subject,
        date_from=date_from,
        date_to=date_to
    )

    return export_to_csv(df, filepath)


def export_attendance_to_excel(
    student_id: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    filepath: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Quick export of attendance records to Excel.

    Args:
        student_id: Filter by student
        subject: Filter by subject
        date_from: Start date
        date_to: End date
        filepath: Output file path

    Returns:
        Tuple of (success, message)
    """
    generator = ReportGenerator()
    df = generator.generate_report(
        student_id=student_id,
        subject=subject,
        date_from=date_from,
        date_to=date_to
    )

    return export_to_excel(df, filepath)


# =============================================================================
# REPORT STATISTICS
# =============================================================================

def get_report_statistics(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate statistics from a report DataFrame.

    Args:
        df: pandas DataFrame with attendance records

    Returns:
        Dict with statistics
    """
    if df.empty:
        return {'total_records': 0}

    stats = {
        'total_records': len(df),
        'unique_students': df['student_id'].nunique() if 'student_id' in df.columns else 0,
        'unique_subjects': df['subject'].nunique() if 'subject' in df.columns else 0,
        'unique_dates': df['date'].nunique() if 'date' in df.columns else 0
    }

    # Attendance rate
    if 'status' in df.columns:
        present_count = (df['status'] == 'Present').sum()
        stats['present_count'] = present_count
        stats['attendance_rate'] = present_count / len(df) * 100

    # Subject-wise breakdown
    if 'subject' in df.columns:
        stats['by_subject'] = df.groupby('subject').size().to_dict()

    # Section-wise breakdown
    if 'section' in df.columns:
        stats['by_section'] = df.groupby('section').size().to_dict()

    return stats


def generate_low_attendance_list(
    threshold: float = 75.0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate list of students with attendance below threshold.

    Args:
        threshold: Minimum attendance percentage
        date_from: Start date for filtering
        date_to: End date for filtering

    Returns:
        DataFrame with students below threshold
    """
    students = get_all_students()
    low_attendance = []

    for student in students:
        summary = get_student_attendance_summary(student['student_id'])

        if summary['percentage'] < threshold and summary['total_classes'] > 0:
            low_attendance.append({
                'student_id': student['student_id'],
                'name': student['name'],
                'department': student['department'],
                'section': student['section'],
                'total_classes': summary['total_classes'],
                'present_count': summary['present_count'],
                'percentage': summary['percentage'],
                'deficit': threshold - summary['percentage']
            })

    df = pd.DataFrame(low_attendance)

    if not df.empty:
        df = df.sort_values('percentage')

    return df
