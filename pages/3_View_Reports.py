"""
Filterable attendance reports with CSV and Excel export,
plus student and subject summaries.
"""

import io
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import config
from src.database import create_tables, get_all_students, get_attendance_records

st.set_page_config(
    page_title="View Reports",
    layout="wide",
    initial_sidebar_state="collapsed",
)
create_tables()


def render_filters() -> dict:
    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    today = datetime.now()

    with c1:
        preset = st.selectbox(
            "Date range",
            ["Today", "Yesterday", "Last 7 days", "Last 30 days", "Custom", "All time"],
        )
        if preset == "Today":
            d0 = d1 = today.strftime(config.DATE_FORMAT)
        elif preset == "Yesterday":
            yd = today - timedelta(days=1)
            d0 = d1 = yd.strftime(config.DATE_FORMAT)
        elif preset == "Last 7 days":
            d0 = (today - timedelta(days=7)).strftime(config.DATE_FORMAT)
            d1 = today.strftime(config.DATE_FORMAT)
        elif preset == "Last 30 days":
            d0 = (today - timedelta(days=30)).strftime(config.DATE_FORMAT)
            d1 = today.strftime(config.DATE_FORMAT)
        elif preset == "Custom":
            ca, cb = st.columns(2)
            d0 = ca.date_input("From", today - timedelta(days=7)).strftime(config.DATE_FORMAT)
            d1 = cb.date_input("To", today).strftime(config.DATE_FORMAT)
        else:
            d0 = d1 = None

    with c2:
        subjects = ["All"] + config.DEFAULT_SUBJECTS
        subj = st.selectbox("Subject", subjects)
        sections = ["All"] + config.SECTIONS
        sec = st.selectbox("Section", sections)

    with c3:
        students = get_all_students()
        options = {"All": None, **{f"{s['name']} ({s['student_id']})": s["student_id"] for s in students}}
        chosen = st.selectbox("Student", list(options.keys()))
        statuses = ["All", "Present", "Absent", "Late"]
        status = st.selectbox("Status", statuses)

    return dict(
        date_from=d0,
        date_to=d1,
        subject=None if subj == "All" else subj,
        section=None if sec == "All" else sec,
        student_id=options[chosen],
        status=None if status == "All" else status,
    )


def render_table(records: list) -> pd.DataFrame | None:
    if not records:
        st.info("No records found for the selected filters.")
        return None

    df = pd.DataFrame(records).rename(
        columns={
            "student_id": "ID",
            "name": "Name",
            "subject": "Subject",
            "section": "Section",
            "date": "Date",
            "time": "Time",
            "status": "Status",
        }
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", len(df))
    present = (df["Status"] == "Present").sum() if "Status" in df.columns else 0
    c2.metric("Present", int(present))
    c3.metric("Absent", int(len(df) - present))
    c4.metric("Rate", f"{present / len(df) * 100:.1f}%" if len(df) else "—")

    st.dataframe(df, use_container_width=True, hide_index=True)
    return df


def render_export(df: pd.DataFrame | None):
    if df is None or df.empty:
        return
    st.subheader("Export")
    c1, c2 = st.columns(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with c1:
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name=f"attendance_{ts}.csv", mime="text/csv")

    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")
        st.download_button(
            "Download Excel",
            buf.getvalue(),
            file_name=f"attendance_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_student_summary():
    st.subheader("Student summary")
    c1, c2 = st.columns(2)
    d0 = c1.date_input("From", datetime.now() - timedelta(days=30), key="sw_from")
    d1 = c2.date_input("To", datetime.now(), key="sw_to")

    if not st.button("Generate student report"):
        return

    rows = []
    for student in get_all_students():
        recs = get_attendance_records(
            student_id=student["student_id"],
            date_from=d0.strftime(config.DATE_FORMAT),
            date_to=d1.strftime(config.DATE_FORMAT),
        )
        total = len(recs)
        present = sum(1 for r in recs if r.get("status") == "Present")
        rows.append(
            {
                "ID": student["student_id"],
                "Name": student["name"],
                "Dept": student["department"],
                "Section": student["section"],
                "Total": total,
                "Present": present,
                "Rate %": round(present / total * 100, 1) if total else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    low = df[df["Rate %"] < 75]
    if not low.empty:
        st.warning(f"{len(low)} student(s) are below 75% attendance")
        with st.expander("View students below 75%"):
            st.dataframe(low[["ID", "Name", "Rate %"]], use_container_width=True)


def render_subject_summary():
    st.subheader("Subject summary")
    c1, c2 = st.columns(2)
    d0 = c1.date_input("From", datetime.now() - timedelta(days=30), key="subj_from")
    d1 = c2.date_input("To", datetime.now(), key="subj_to")

    if not st.button("Generate subject report"):
        return

    recs = get_attendance_records(
        date_from=d0.strftime(config.DATE_FORMAT),
        date_to=d1.strftime(config.DATE_FORMAT),
    )
    if not recs:
        st.info("No records found.")
        return

    df = pd.DataFrame(recs)
    rows = []
    for subject, grp in df.groupby("subject"):
        total = len(grp)
        present = (grp["status"] == "Present").sum()
        rows.append(
            {
                "Subject": subject,
                "Students": grp["student_id"].nunique(),
                "Records": total,
                "Present": int(present),
                "Rate %": round(present / total * 100, 1) if total else 0.0,
            }
        )
    result = pd.DataFrame(rows).sort_values("Rate %", ascending=False)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.bar_chart(result.set_index("Subject")["Rate %"])


def main():
    st.title("View Reports")
    st.markdown("---")

    filters = render_filters()
    st.markdown("---")

    records = get_attendance_records(
        date_from=filters["date_from"],
        date_to=filters["date_to"],
        subject=filters["subject"],
        section=filters["section"],
        student_id=filters["student_id"],
    )
    if filters["status"]:
        records = [r for r in records if r.get("status") == filters["status"]]

    df = render_table(records)
    render_export(df)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        render_student_summary()
    with col2:
        render_subject_summary()


if __name__ == "__main__":
    main()
