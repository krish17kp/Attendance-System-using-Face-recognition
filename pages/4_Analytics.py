"""
Attendance dashboard with trends, breakdowns, and performance summaries.
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import config
from src.database import create_tables, get_all_students, get_attendance_records

st.set_page_config(
    page_title="Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)
create_tables()


def load_data(days: int = 30):
    today = datetime.now()
    start = (today - timedelta(days=days)).strftime(config.DATE_FORMAT)
    end = today.strftime(config.DATE_FORMAT)
    return get_attendance_records(date_from=start, date_to=end)


def render_overview():
    students = get_all_students()
    records = load_data(30)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", len(students))
    c2.metric("Records (last 30 days)", len(records))

    if records:
        df = pd.DataFrame(records)
        present = (df["status"] == "Present").sum()
        rate = round(present / len(df) * 100, 1)
        dates = df["date"].nunique()
    else:
        rate = 0.0
        dates = 0

    c3.metric("Overall attendance rate", f"{rate}%")
    c4.metric("Active days", dates)


def render_trends():
    st.subheader("Attendance trends")
    period = st.selectbox("Period", ["Last 7 days", "Last 30 days", "Last 90 days"], index=1, key="trend_period")
    days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[period]

    records = load_data(days)
    if not records:
        st.info("No data for this period.")
        return

    df = pd.DataFrame(records)
    daily = (
        df.groupby("date")
        .agg(total=("student_id", "count"), present=("status", lambda x: (x == "Present").sum()))
        .reset_index()
    )
    daily["rate"] = (daily["present"] / daily["total"] * 100).round(1)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    st.markdown("**Daily attendance rate (%)**")
    st.line_chart(daily.set_index("date")[["rate"]])

    st.markdown("**Present vs. absent per day**")
    daily["absent"] = daily["total"] - daily["present"]
    st.bar_chart(daily.set_index("date")[["present", "absent"]])


def render_dept_section():
    records = load_data(30)
    students = get_all_students()
    if not records or not students:
        st.info("Not enough data.")
        return

    rdf = pd.DataFrame(records)
    sdf = pd.DataFrame(students)[["student_id", "department"]]
    merged = rdf.merge(sdf, on="student_id", how="left")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("By department")
        dept = (
            merged.groupby("department")
            .agg(total=("student_id", "count"), present=("status", lambda x: (x == "Present").sum()))
            .reset_index()
        )
        dept["rate"] = (dept["present"] / dept["total"] * 100).round(1)
        st.dataframe(dept.rename(columns={"department": "Department", "rate": "Rate %"}), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("By section")
        sec = (
            rdf.groupby("section")
            .agg(total=("student_id", "count"), present=("status", lambda x: (x == "Present").sum()))
            .reset_index()
        )
        sec["rate"] = (sec["present"] / sec["total"] * 100).round(1)
        st.dataframe(sec.rename(columns={"section": "Section", "rate": "Rate %"}), use_container_width=True, hide_index=True)


def render_subjects():
    st.subheader("Subject analysis")
    records = load_data(30)
    if not records:
        st.info("No data.")
        return

    df = pd.DataFrame(records)
    subj = (
        df.groupby("subject")
        .agg(total=("student_id", "count"), present=("status", lambda x: (x == "Present").sum()))
        .reset_index()
    )
    subj["rate"] = (subj["present"] / subj["total"] * 100).round(1)
    subj = subj.sort_values("rate", ascending=False)

    c1, c2 = st.columns([2, 1])
    c1.bar_chart(subj.set_index("subject")["rate"])
    c2.dataframe(subj.rename(columns={"subject": "Subject", "rate": "Rate %"}), use_container_width=True, hide_index=True)


def render_student_performance():
    st.subheader("Student performance")
    records = load_data(30)
    students = get_all_students()
    if not records or not students:
        st.info("Not enough data.")
        return

    rdf = pd.DataFrame(records)
    sdf = pd.DataFrame(students)[["student_id", "name", "department"]]
    perf = (
        rdf.groupby("student_id")
        .agg(total=("status", "count"), present=("status", lambda x: (x == "Present").sum()))
        .reset_index()
    )
    perf["rate"] = (perf["present"] / perf["total"] * 100).round(1)
    perf = perf.merge(sdf, on="student_id", how="left")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top performers**")
        top = perf.nlargest(10, "rate")[["name", "department", "rate"]]
        st.dataframe(top.rename(columns={"name": "Name", "department": "Dept", "rate": "Rate %"}), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Below 75% attendance**")
        low = perf[perf["rate"] < 75].nsmallest(10, "rate")[["name", "department", "rate"]]
        if low.empty:
            st.success("All students are at or above 75%.")
        else:
            st.dataframe(low.rename(columns={"name": "Name", "department": "Dept", "rate": "Rate %"}), use_container_width=True, hide_index=True)


def render_summary():
    records = load_data(30)
    if not records:
        return
    st.subheader("Summary")
    df = pd.DataFrame(records)
    rate = round((df["status"] == "Present").mean() * 100, 1)
    st.markdown(f"- Overall attendance rate for the last 30 days: **{rate}%**")

    best_day = df.groupby("date")["status"].apply(lambda x: (x == "Present").mean()).idxmax()
    best_rate = df.groupby("date")["status"].apply(lambda x: (x == "Present").mean()).max()
    st.markdown(f"- Best day: **{best_day}** ({best_rate * 100:.1f}%)")

    unique = df["student_id"].nunique()
    st.markdown(f"- Unique students marked in this period: **{unique}**")


def main():
    st.title("Analytics")
    st.markdown("---")
    render_overview()

    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Department / Section", "Subjects", "Students"])
    with tab1:
        render_trends()
    with tab2:
        render_dept_section()
    with tab3:
        render_subjects()
    with tab4:
        render_student_performance()

    st.markdown("---")
    render_summary()


if __name__ == "__main__":
    main()
