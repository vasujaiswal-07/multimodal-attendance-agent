import csv
from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from face_utils import (
    ensure_data, read_students, add_student, save_face_photo,
    train_model, recognize_frame, STUDENTS, FACES, DATA
)
from audio_analysis import analyze_wav, suggestions

ensure_data()
ATTENDANCE = DATA / "attendance.csv"
AUDIO_REPORTS = DATA / "audio_reports.csv"

st.set_page_config(page_title="Multimodal Attendance Agent", page_icon="🎓", layout="wide")
st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Offline classroom prototype • Local data storage • Human review required")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose module", [
        "Overview", "Student Enrollment", "Live Attendance",
        "Audio Analysis", "Session Report"
    ])
    st.divider()
    st.info("Privacy: collect data only with informed consent. Provide a non-biometric alternative.")

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    students = read_students()
    c1.metric("Registered students", len(students))
    c2.metric("Attendance records", len(pd.read_csv(ATTENDANCE)) if ATTENDANCE.exists() and ATTENDANCE.stat().st_size else 0)
    c3.metric("Audio reports", len(pd.read_csv(AUDIO_REPORTS)) if AUDIO_REPORTS.exists() and AUDIO_REPORTS.stat().st_size else 0)
    st.markdown("""
    This prototype combines:
    - **Computer vision:** face detection and local LBPH face recognition.
    - **Audio signal analysis:** broad sound activity and low-activity intervals.
    - **Agent-style reporting:** rule-based suggestions based on session indicators.
    - **Analytics:** attendance and audio summaries.

    Audio activity is not the same as engagement, and face visibility is not proof of attention.
    """)
    st.subheader("Quick start")
    st.write("1. Enroll students with consented photos.")
    st.write("2. Train the face model.")
    st.write("3. Start webcam attendance.")
    st.write("4. Upload a WAV recording for audio analysis.")
    st.write("5. Review the session report.")

elif page == "Student Enrollment":
    st.subheader("Register a student")
    with st.form("student_form"):
        sid = st.text_input("Student ID", placeholder="e.g. STU001")
        name = st.text_input("Student full name")
        submitted = st.form_submit_button("Register student")
    if submitted:
        try:
            add_student(sid, name)
            st.success(f"Registered {name}. Now upload face photos below.")
        except Exception as e:
            st.error(str(e))

    students = read_students()
    if students:
        st.subheader("Upload face photos")
        choices = {f'{s["student_name"]} — {s["student_id"]}': s["student_id"] for s in students}
        selected = st.selectbox("Choose registered student", list(choices))
        uploaded = st.file_uploader("Upload one or more clear face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button("Save uploaded photos", disabled=not uploaded):
            success, errors = 0, []
            for idx, item in enumerate(uploaded, start=1):
                try:
                    pil = Image.open(item).convert("RGB")
                    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    save_face_photo(choices[selected], bgr, datetime.now().strftime("%Y%m%d%H%M%S") + str(idx))
                    success += 1
                except Exception as e:
                    errors.append(f"{item.name}: {e}")
            st.success(f"Saved {success} face photo(s).")
            for err in errors:
                st.warning(err)
        if st.button("Train / retrain face model"):
            try:
                n_images, n_students = train_model()
                st.success(f"Model trained using {n_images} photos from {n_students} student(s).")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Register a student first.")

elif page == "Live Attendance":
    st.subheader("Webcam attendance")
    st.warning("Use only with consent. Review every record before treating it as official attendance.")
    threshold = st.slider("Recognition strictness (lower is stricter)", 35, 100, 75)
    duration = st.slider("Scan duration in seconds", 5, 60, 15)
    st.caption("The scan records each recognized student once during this scan. Make sure lighting is good.")
    if st.button("Start webcam scan"):
        try:
            # Check that a model exists before opening the camera.
            from face_utils import load_model
            load_model()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Check camera permission or close other camera apps.")
            else:
                frame_box = st.empty()
                status_box = st.empty()
                seen = {}
                start = datetime.now()
                while (datetime.now() - start).total_seconds() < duration:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    try:
                        found = recognize_frame(frame, threshold=threshold)
                    except Exception as e:
                        st.error(str(e))
                        break
                    for x, y, w, h, name, student_id, confidence in found:
                        color = (0, 180, 0) if student_id else (0, 0, 220)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, name, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if student_id:
                            seen[student_id] = name.split(" (")[0]
                    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    status_box.write(f"Recognized {len(seen)} unique student(s).")
                cap.release()
                if seen:
                    new_rows = []
                    now = datetime.now()
                    for student_id, student_name in seen.items():
                        new_rows.append({
                            "date": now.strftime("%Y-%m-%d"),
                            "time": now.strftime("%H:%M:%S"),
                            "student_id": student_id,
                            "student_name": student_name,
                            "status": "Present",
                            "source": "webcam"
                        })
                    df_new = pd.DataFrame(new_rows)
                    if ATTENDANCE.exists() and ATTENDANCE.stat().st_size:
                        old = pd.read_csv(ATTENDANCE)
                        combined = pd.concat([old, df_new], ignore_index=True)
                    else:
                        combined = df_new
                    combined = combined.drop_duplicates(subset=["date", "student_id"], keep="last")
                    combined.to_csv(ATTENDANCE, index=False)
                    st.success(f"Saved attendance for {len(seen)} student(s).")
                    st.dataframe(df_new, use_container_width=True)
                else:
                    st.warning("No known students recognized. Check photos, lighting, and model training.")
        except Exception as e:
            st.error(f"Could not start attendance: {e}")

    if ATTENDANCE.exists() and ATTENDANCE.stat().st_size:
        st.subheader("Recent attendance")
        st.dataframe(pd.read_csv(ATTENDANCE).tail(30), use_container_width=True)
    else:
        st.info("No attendance records yet.")

elif page == "Audio Analysis":
    st.subheader("Classroom audio activity analysis")
    st.write("Upload a WAV file. The app calculates broad volume/activity indicators; it does not identify speakers or determine attention.")
    audio_file = st.file_uploader("Choose WAV audio", type=["wav"])
    if audio_file is not None and st.button("Analyze audio"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(audio_file.getvalue())
                temp_path = temp.name
            report = analyze_wav(temp_path)
            st.session_state["last_audio_report"] = report
            st.metric("Recording duration", f'{report["duration_seconds"]} sec')
            c1, c2, c3 = st.columns(3)
            c1.metric("Sound activity", f'{report["sound_activity_percent"]}%')
            c2.metric("Low-activity intervals", f'{report["low_activity_percent"]}%')
            c3.metric("Mean RMS level", report["mean_rms"])
            st.line_chart(pd.DataFrame({"RMS sound level (0.5-second windows)": report["windows"]}))
            st.subheader("Suggested teaching checks")
            for item in suggestions(report):
                st.write(f"- {item}")
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file_name": audio_file.name,
                "duration_seconds": report["duration_seconds"],
                "sound_activity_percent": report["sound_activity_percent"],
                "low_activity_percent": report["low_activity_percent"],
                "mean_rms": report["mean_rms"]
            }
            if AUDIO_REPORTS.exists() and AUDIO_REPORTS.stat().st_size:
                old = pd.read_csv(AUDIO_REPORTS)
                new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            else:
                new = pd.DataFrame([row])
            new.to_csv(AUDIO_REPORTS, index=False)
            st.success("Audio summary saved locally.")
        except Exception as e:
            st.error(f"Could not analyze this file: {e}")

elif page == "Session Report":
    st.subheader("Session report")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Attendance records")
        if ATTENDANCE.exists() and ATTENDANCE.stat().st_size:
            att = pd.read_csv(ATTENDANCE)
            st.dataframe(att, use_container_width=True)
            st.download_button("Download attendance CSV", att.to_csv(index=False).encode("utf-8"), "attendance.csv", "text/csv")
        else:
            st.info("No attendance records available.")
    with right:
        st.markdown("#### Audio summaries")
        if AUDIO_REPORTS.exists() and AUDIO_REPORTS.stat().st_size:
            aud = pd.read_csv(AUDIO_REPORTS)
            st.dataframe(aud, use_container_width=True)
            st.download_button("Download audio reports CSV", aud.to_csv(index=False).encode("utf-8"), "audio_reports.csv", "text/csv")
            avg = aud["sound_activity_percent"].mean()
            st.write(f"Average recorded sound activity: **{avg:.1f}%**")
            st.caption("This is a sound-level statistic, not a student engagement score.")
        else:
            st.info("No audio reports available.")
    st.subheader("Agent-generated teaching suggestions")
    st.write("Use suggestions as prompts for reflection, not as automatic judgments.")
    st.write("- If sound activity is high, check whether it reflects useful discussion or distracting noise.")
    st.write("- If low-activity intervals are frequent, consider a comprehension check when appropriate.")
    st.write("- Compare audio indicators with lesson type, student feedback, and teacher observations.")
    st.write("- Manually verify attendance and allow corrections for recognition errors.")
