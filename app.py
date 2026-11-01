from pathlib import Path
from datetime import datetime
import json
import math
import re
import tempfile
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
NLP_MODEL_PATH = Path(__file__).parent / "driveaware_intent_model"
NLP_MIN_CONFIDENCE = 0.55

st.set_page_config(page_title="DriveAware AI", page_icon="🚗", layout="wide")
if "clip_history" not in st.session_state:
    st.session_state.clip_history = []
if "cue_event_history" not in st.session_state:
    st.session_state.cue_event_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("DriveAware AI")
st.caption("CV prototype with temporal cues, reliability checks, and a grounded NLP assistant")
st.info(
    "This learning prototype displays facial landmarks only. It does not diagnose "
    "fatigue or determine whether someone is safe to drive. Uploaded videos are "
    "processed locally by this app."
)

with st.sidebar:
    st.header("Monitoring status")
    st.write("Camera: not started")
    st.write("Fatigue estimate: not available yet")
    st.write("Current phase: CV + grounded Q&A baseline")

st.subheader("Analyze a recorded video")
st.write(
    "Upload a short video. The app samples a few frames and marks detected face "
    "landmarks with green dots. It measures eye/mouth opening and checks face "
    "tracking coverage and face-region brightness. Start with a short clip (10–30 seconds)."
)
video_file = st.file_uploader("Choose a driver video", type=["mp4", "mov", "avi", "m4v"])
sample_fps = st.slider("Frames to analyze per second", min_value=1, max_value=8, value=5)
eye_threshold = st.slider(
    "Demo threshold for low eye opening (EAR)", min_value=0.10, max_value=0.35,
    value=0.20, step=0.01,
)
mouth_threshold = st.slider(
    "Demo threshold for open mouth (MAR)", min_value=0.10, max_value=0.60,
    value=0.30, step=0.01,
)
st.caption(
    "EAR and MAR are approximate geometry measurements. The thresholds are "
    "starting points for a demo and vary by person, camera angle, and video."
)

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_POINTS = (13, 14, 61, 291)
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def point_distance(landmarks, first, second):
    a, b = landmarks[first], landmarks[second]
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, indexes):
    horizontal = point_distance(landmarks, indexes[0], indexes[3])
    if horizontal == 0:
        return 0.0
    vertical = (
        point_distance(landmarks, indexes[1], indexes[5])
        + point_distance(landmarks, indexes[2], indexes[4])
    ) / 2
    return vertical / horizontal


def measure_cues(landmarks):
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
    mouth_width = point_distance(landmarks, 61, 291)
    mar = point_distance(landmarks, 13, 14) / mouth_width if mouth_width else 0.0
    return ear, mar


def estimate_head_pose(frame_shape, landmarks):
    height, width = frame_shape[:2]
    image_points = np.array(
        [(landmarks[index].x * width, landmarks[index].y * height) for index in POSE_LANDMARKS],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2],
            [0.0, focal_length, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vector, _ = cv2.solvePnP(
        POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return tuple(float(angle) for angle in angles)


def add_relative_pitch(samples, sample_rate):
    valid_samples = [sample for sample in samples if sample["Pitch (deg)"] is not None]
    neutral_count = min(len(valid_samples), max(3, round(sample_rate * 2)))
    if neutral_count == 0:
        return
    baseline_pitch = float(np.median([sample["Pitch (deg)"] for sample in valid_samples[:neutral_count]]))
    for sample in samples:
        pitch = sample["Pitch (deg)"]
        sample["Pitch change (deg)"] = (
            abs((pitch - baseline_pitch + 180) % 360 - 180) if pitch is not None else None
        )


def find_temporal_cues(samples, sample_rate, eye_limit, mouth_limit):
    """Report sustained visual cues; these are not fatigue diagnoses."""
    required_eye_samples = max(2, math.ceil(sample_rate * 1.5) + 1)
    required_mouth_samples = max(2, math.ceil(sample_rate * 1.0) + 1)
    required_pitch_samples = max(2, math.ceil(sample_rate * 1.0) + 1)
    events = []
    eye_start = mouth_start = pitch_start = None
    eye_run = mouth_run = pitch_run = 0
    eye_reported = mouth_reported = pitch_reported = False

    for sample in samples:
        if not sample["Face detected"]:
            eye_start = mouth_start = pitch_start = None
            eye_run = mouth_run = pitch_run = 0
            eye_reported = mouth_reported = pitch_reported = False
            continue

        second = sample["Video time (s)"]
        ear, mar = sample["EAR"], sample["MAR"]
        if ear is not None and ear < eye_limit:
            eye_start = second if eye_start is None else eye_start
            eye_run += 1
            if eye_run >= required_eye_samples and not eye_reported:
                events.append({
                    "Cue": "Sustained low eye opening",
                    "Started (s)": round(eye_start, 2),
                    "Rule met (s)": round(second, 2),
                    "Evidence": f"EAR below {eye_limit:.2f} for about {second - eye_start:.1f} s",
                })
                eye_reported = True
        else:
            eye_start, eye_run, eye_reported = None, 0, False

        if mar is not None and mar > mouth_limit:
            mouth_start = second if mouth_start is None else mouth_start
            mouth_run += 1
            if mouth_run >= required_mouth_samples and not mouth_reported:
                events.append({
                    "Cue": "Sustained mouth opening",
                    "Started (s)": round(mouth_start, 2),
                    "Rule met (s)": round(second, 2),
                    "Evidence": f"MAR above {mouth_limit:.2f} for about {second - mouth_start:.1f} s; speech can also cause this",
                })
                mouth_reported = True
        else:
            mouth_start, mouth_run, mouth_reported = None, 0, False

        pitch_change = sample["Pitch change (deg)"]
        if pitch_change is not None and pitch_change >= 20:
            pitch_start = second if pitch_start is None else pitch_start
            pitch_run += 1
            if pitch_run >= required_pitch_samples and not pitch_reported:
                events.append({
                    "Cue": "Sustained head-pitch change",
                    "Started (s)": round(pitch_start, 2),
                    "Rule met (s)": round(second, 2),
                    "Evidence": f"Pitch changed by at least 20° from the initial pose for about {second - pitch_start:.1f} s",
                })
                pitch_reported = True
        else:
            pitch_start, pitch_run, pitch_reported = None, 0, False

    return events


def classify_question(question):
    """Small transparent NLP baseline using normalized words and intent cues."""
    normalized = question.lower().strip()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))

    if tokens.intersection({"camera", "lighting", "brightness", "visible", "visibility", "blocked", "working"}):
        return "camera_status"
    if tokens.intersection({"why", "reason", "trigger", "triggered", "cause", "alert"}):
        return "why_cue"
    if tokens.intersection({"recent", "events", "event", "detected", "happened", "history", "cues"}):
        return "recent_events"
    if tokens.intersection({"summary", "summarize", "overall", "session", "totals"}):
        return "session_summary"
    if tokens.intersection({"help", "questions", "ask"}):
        return "help"
    return "unsupported"


@st.cache_resource(show_spinner=False)
def load_intent_model():
    """Load the optional local DistilBERT checkpoint for experimental use."""
    if not (NLP_MODEL_PATH / "config.json").exists():
        return None, "No local DistilBERT checkpoint was found."
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(NLP_MODEL_PATH), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(NLP_MODEL_PATH), local_files_only=True
        )
        model.to(torch.device("cpu"))
        model.eval()
        return {"torch": torch, "tokenizer": tokenizer, "model": model}, None
    except Exception as error:
        return None, str(error)


def classify_with_intent_model(question, bundle):
    torch = bundle["torch"]
    encoded = bundle["tokenizer"](
        question, truncation=True, padding="max_length", max_length=64, return_tensors="pt"
    )
    with torch.inference_mode():
        probabilities = torch.softmax(bundle["model"](**encoded).logits, dim=-1)[0]
    confidence, label_id = torch.max(probabilities, dim=-1)
    intent = bundle["model"].config.id2label[int(label_id.item())]
    return intent, float(confidence.item())


def read_intent_test_metrics():
    metrics_path = NLP_MODEL_PATH / "evaluation.json"
    if not metrics_path.exists():
        return None
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = report.get("test", {})
        return test_metrics.get("distilbert", test_metrics)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def answer_from_session(intent):
    clips = st.session_state.clip_history
    events = st.session_state.cue_event_history
    if intent == "why_cue":
        if events:
            event = events[-1]
            return (
                f"In **{event['Video']}**, the system recorded **{event['Cue']}**. "
                f"It began at about {event['Started (s)']} s and met the rule at "
                f"{event['Rule met (s)']} s because {event['Evidence']}. This is a "
                "measured visual cue, not a fatigue diagnosis."
            )
        if clips:
            return "No sustained cue was recorded in this session. The system only reports a cue after its time rule is met and visual evidence passes the quality check."
        return "No video has been analyzed in this app session yet, so I do not have an event to explain."

    if intent == "recent_events":
        if not events:
            return "There are no sustained cues in the current session history. Analyze a video first; short-lived measurements may not become events."
        lines = [
            f"- **{event['Cue']}** in {event['Video']} at {event['Started (s)']} s: {event['Evidence']}"
            for event in events[-5:]
        ]
        return "Recent recorded cues:\n\n" + "\n".join(lines)

    if intent == "session_summary":
        if not clips:
            return "No video has been analyzed in this app session yet, so there is no session summary to report."
        coverage = np.mean([float(clip["Face coverage"].rstrip("%")) for clip in clips])
        return (
            f"This app session contains **{len(clips)} analyzed clip(s)** and "
            f"**{len(events)} sustained cue(s)**. Average face-tracking coverage was "
            f"{coverage:.0f}%. These are visual observations from the clip analyses, "
            "not a diagnosis of driver fatigue."
        )

    if intent == "camera_status":
        if not clips:
            return "No video has been analyzed yet, so camera visibility and brightness are unavailable."
        latest = clips[-1]
        return (
            f"For the latest clip (**{latest['Video']}**), face-tracking coverage was "
            f"{latest['Face coverage']} and average face-region brightness was "
            f"{latest['Face brightness']}/255. The visual-evidence status was "
            f"**{latest['Visual evidence']}**. Coverage and brightness are simple "
            "quality checks, not a complete camera-health test."
        )

    if intent == "help":
        return "You can ask why a cue was recorded, what recent events were detected, for a session summary, or whether the latest clip had adequate visual evidence."

    return "I can answer questions about recorded cues, recent events, the session summary, and visual-evidence quality. Try rephrasing using one of those topics."


def get_face_landmarker():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading the face landmark model (first run only)…"):
            urlretrieve(MODEL_URL, MODEL_PATH)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


if video_file is not None and st.button("Analyze video", type="primary"):
    suffix = Path(video_file.name).suffix or ".mp4"
    temporary_video = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_file.getvalue())
            temporary_video = temp_file.name

        capture = cv2.VideoCapture(temporary_video)
        if not capture.isOpened():
            st.error("I couldn't open this video. Try an MP4 file encoded with H.264.")
        else:
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 25.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(1, round(fps / sample_fps))
            estimated_samples = max(1, min(180, (frame_count + frame_step - 1) // frame_step))
            progress = st.progress(0, text="Preparing face tracking…")
            preview_slot = st.empty()
            face_landmarker = get_face_landmarker()
            sampled = 0
            detected = 0
            preview = None
            measurements = []
            face_brightness_values = []

            try:
                frame_number = 0
                while sampled < 180:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    if frame_number % frame_step == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb_frame),
                        )
                        result = face_landmarker.detect(mp_image)
                        sampled += 1
                        ear = mar = None
                        pitch = yaw = roll = None
                        face_brightness = None

                        if result.face_landmarks:
                            detected += 1
                            landmarks = result.face_landmarks[0]
                            ear, mar = measure_cues(landmarks)
                            pose = estimate_head_pose(frame.shape, landmarks)
                            if pose is not None:
                                pitch, yaw, roll = pose
                            height, width = frame.shape[:2]
                            landmark_xs = [point.x for point in landmarks]
                            landmark_ys = [point.y for point in landmarks]
                            x_min = max(0, int(min(landmark_xs) * width))
                            x_max = min(width, int(max(landmark_xs) * width))
                            y_min = max(0, int(min(landmark_ys) * height))
                            y_max = min(height, int(max(landmark_ys) * height))
                            if x_max > x_min and y_max > y_min:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                face_brightness = float(np.mean(gray[y_min:y_max, x_min:x_max]))
                                face_brightness_values.append(face_brightness)
                            for point in landmarks:
                                x = min(max(int(point.x * width), 0), width - 1)
                                y = min(max(int(point.y * height), 0), height - 1)
                                cv2.circle(frame, (x, y), 1, (60, 220, 90), -1)
                            for index in (*LEFT_EYE, *RIGHT_EYE, *MOUTH_POINTS):
                                point = landmarks[index]
                                x = min(max(int(point.x * width), 0), width - 1)
                                y = min(max(int(point.y * height), 0), height - 1)
                                cv2.circle(frame, (x, y), 3, (0, 220, 255), -1)
                            cv2.putText(
                                frame,
                                f"EAR {ear:.2f}  MAR {mar:.2f}  PITCH {pitch:.0f}" if pitch is not None else f"EAR {ear:.2f}  MAR {mar:.2f}",
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
                            )

                        measurements.append(
                            {
                                "Sample": sampled,
                                "Video time (s)": frame_number / fps,
                                "EAR": ear,
                                "MAR": mar,
                                "Pitch (deg)": pitch,
                                "Yaw (deg)": yaw,
                                "Roll (deg)": roll,
                                "Pitch change (deg)": None,
                                "Face detected": ear is not None,
                                "Face brightness": face_brightness,
                                "Low eye opening": ear is not None and ear < eye_threshold,
                                "Open mouth": mar is not None and mar > mouth_threshold,
                            }
                        )

                        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if sampled % 10 == 0 or sampled == 1:
                            preview_slot.image(preview, caption="Processing preview", use_container_width=True)
                        progress.progress(
                            min(sampled / estimated_samples, 1.0),
                            text=f"Analyzed {sampled} sampled frames…",
                        )
                    frame_number += 1
            finally:
                face_landmarker.close()
                capture.release()

            add_relative_pitch(measurements, sample_fps)
            if preview is not None:
                preview_slot.image(preview, caption="Last analyzed frame (green dots = face landmarks)", use_container_width=True)
                st.subheader("Face tracking result")
                st.metric("Frames analyzed", sampled)
                st.metric("Frames with a face detected", detected)
                face_coverage = detected / sampled if sampled else 0.0
                average_brightness = float(np.mean(face_brightness_values)) if face_brightness_values else 0.0
                quality_adequate = face_coverage >= 0.70 and average_brightness >= 45
                analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                valid_samples = [sample for sample in measurements if sample["EAR"] is not None]
                average_ear = float(np.mean([sample["EAR"] for sample in valid_samples])) if valid_samples else None
                average_mar = float(np.mean([sample["MAR"] for sample in valid_samples])) if valid_samples else None
                temporal_events = (
                    find_temporal_cues(measurements, sample_fps, eye_threshold, mouth_threshold)
                    if quality_adequate
                    else []
                )
                st.session_state.clip_history.append(
                    {
                        "Video": video_file.name,
                        "Analyzed at": analyzed_at,
                        "Frames": sampled,
                        "Face coverage": f"{face_coverage:.0%}",
                        "Face brightness": round(average_brightness, 1) if face_brightness_values else None,
                        "Visual evidence": "Adequate" if quality_adequate else "Insufficient",
                        "Average EAR": round(average_ear, 3) if average_ear is not None else None,
                        "Average MAR": round(average_mar, 3) if average_mar is not None else None,
                        "Sustained cues": len(temporal_events),
                    }
                )
                for event in temporal_events:
                    st.session_state.cue_event_history.append(
                        {"Video": video_file.name, "Analyzed at": analyzed_at, **event}
                    )
                quality_columns = st.columns(3)
                quality_columns[0].metric("Face tracking coverage", f"{face_coverage:.0%}")
                quality_columns[1].metric(
                    "Average face brightness", f"{average_brightness:.0f}/255" if face_brightness_values else "Unavailable"
                )
                quality_columns[2].metric(
                    "Visual evidence", "Adequate for demo" if quality_adequate else "Insufficient"
                )
                st.caption(
                    "Coverage is the share of sampled frames where a face was tracked. "
                    "Low coverage can result from framing, occlusion, pose, or poor visibility. "
                    "Brightness is a simple grayscale measure, not a full image-quality score."
                )
                if detected == 0:
                    st.warning("No face was detected. Try a brighter clip with the face centered and unobstructed.")
                else:
                    st.success("Face landmarks and eye/mouth opening measurements are available.")
                    results = pd.DataFrame(measurements)
                    valid = results.dropna(subset=["EAR", "MAR"])
                    low_eye = int(valid["Low eye opening"].sum())
                    open_mouth = int(valid["Open mouth"].sum())
                    first, second, third, fourth = st.columns(4)
                    first.metric("Average EAR", f"{valid['EAR'].mean():.3f}")
                    second.metric("Low-eye samples", f"{low_eye}/{len(valid)}")
                    third.metric("Average MAR", f"{valid['MAR'].mean():.3f}")
                    fourth.metric("Open-mouth samples", f"{open_mouth}/{len(valid)}")
                    st.write("Measurements across sampled frames")
                    st.line_chart(results.set_index("Sample")[["EAR", "MAR"]])
                    st.write("Head-pitch change relative to the initial pose (degrees)")
                    st.line_chart(results.set_index("Sample")[["Pitch change (deg)"]])
                    st.caption(
                        "A low-eye sample is not automatically a blink or fatigue event. "
                        "A high-mouth sample may also be speech. Head pitch uses the first "
                        "2 seconds as a rough baseline, so camera angle and initial posture matter."
                    )
                    st.subheader("Sustained cue review")
                    st.write(
                        "A cue is listed only after it stays beyond its threshold across "
                        "consecutive sampled frames. Frames without a detected face break the run."
                    )
                    if not quality_adequate:
                        st.warning(
                            "Insufficient visual evidence: sustained-cue results are withheld. "
                            "Improve lighting and keep the face visible and centered, then analyze again."
                        )
                    else:
                        if temporal_events:
                            st.warning("Sustained visual cue(s) found. Review the video context; this is not a fatigue diagnosis.")
                            st.dataframe(pd.DataFrame(temporal_events), use_container_width=True, hide_index=True)
                        else:
                            st.success("No sustained eye, mouth, or head-pitch cue was found in the analyzed samples.")
                    st.caption(
                        "The demo rules use about 1.5 seconds for low eye opening and "
                        "1 second for sustained mouth opening or a 20° pitch change. "
                        "A short blink should not meet the eye rule."
                    )
            else:
                st.warning("No readable frames were found in this video.")
    except Exception as error:
        st.error(f"Video analysis could not start: {error}")
        st.caption("Check your internet connection for the first-run model download, then try again.")
    finally:
        if temporary_video and Path(temporary_video).exists():
            Path(temporary_video).unlink()


st.divider()
st.header("Session summary and event history")
if st.session_state.clip_history:
    history = pd.DataFrame(st.session_state.clip_history)
    average_coverage = np.mean(
        [float(value.rstrip("%")) for value in history["Face coverage"]]
    )
    total_cues = int(history["Sustained cues"].sum())
    summary_columns = st.columns(3)
    summary_columns[0].metric("Clips analyzed this session", len(history))
    summary_columns[1].metric("Average face coverage", f"{average_coverage:.0f}%")
    summary_columns[2].metric("Sustained cues recorded", total_cues)
    st.dataframe(history, use_container_width=True, hide_index=True)

    if st.session_state.cue_event_history:
        st.subheader("Event details")
        st.dataframe(
            pd.DataFrame(st.session_state.cue_event_history),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No sustained cues have been recorded in this app session.")

    st.download_button(
        "Download session summary (CSV)",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="driveaware_session_summary.csv",
        mime="text/csv",
    )
    st.caption("This history is kept in the open app session. It clears when the session resets or you clear it below.")
    if st.button("Clear session history"):
        st.session_state.clip_history = []
        st.session_state.cue_event_history = []
        st.session_state.chat_history = []
        st.rerun()
else:
    st.write("Analyze a video to start this session's history.")


st.divider()
st.header("Ask DriveAware about this session")
st.write(
    "Answers are built only from the current session's recorded events and quality "
    "checks. The keyword baseline is the default; the fine-tuned DistilBERT option is experimental."
)
if (NLP_MODEL_PATH / "config.json").exists():
    model_test_metrics = read_intent_test_metrics()
    test_summary = (
        f"test accuracy {model_test_metrics['accuracy']:.2f}, macro-F1 {model_test_metrics['macro_f1']:.2f}"
        if model_test_metrics and "accuracy" in model_test_metrics and "macro_f1" in model_test_metrics
        else "test metrics unavailable"
    )
    use_distilbert = st.checkbox(
        f"Use experimental DistilBERT intent classifier ({test_summary})",
        value=False,
        help="This score is from a small synthetic held-out question set and may not represent real users.",
    )
    st.caption(
        f"The model is experimental; {test_summary} on a small synthetic set. "
        "the keyword baseline remains the default. A score below 0.55 falls back to the baseline."
    )
else:
    use_distilbert = False
    st.caption(
        "To enable DistilBERT, extract the trained model folder next to app.py and "
        "install the optional dependencies from requirements-nlp.txt."
    )
question = st.chat_input("Try: Why was a cue recorded? or Summarize this session")
if question:
    intent_method = "keyword baseline"
    confidence = None
    if use_distilbert:
        model_bundle, model_error = load_intent_model()
        if model_error:
            st.warning(f"DistilBERT could not be loaded; using the keyword baseline. Details: {model_error}")
            intent = classify_question(question)
            intent_method = "keyword baseline (model unavailable)"
        else:
            predicted_intent, confidence = classify_with_intent_model(question, model_bundle)
            if confidence < NLP_MIN_CONFIDENCE:
                intent = classify_question(question)
                intent_method = "keyword baseline (low model confidence)"
            else:
                intent = predicted_intent
                intent_method = "experimental DistilBERT"
    else:
        intent = classify_question(question)
    answer = answer_from_session(intent)
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "intent": intent,
            "method": intent_method,
            "confidence": confidence,
        }
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            method = message.get("method", "keyword baseline")
            confidence = message.get("confidence")
            confidence_text = f" · confidence {confidence:.2f}" if confidence is not None else ""
            st.caption(
                f"Intent: {message['intent'].replace('_', ' ')} · method: {method}{confidence_text}"
            )
