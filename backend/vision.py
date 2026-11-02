"""Video analysis functions independent of the user interface."""

from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
POSE_IDS = (1, 152, 33, 263, 61, 291)
POSE_POINTS = np.array([
    (0, 0, 0), (0, -330, -65), (-225, 170, -135),
    (225, 170, -135), (-150, -150, -125), (150, -150, -125),
], dtype=np.float64)
_LIVE_DETECT_LOCK = Lock()


def distance(points, a, b):
    return math.hypot(points[a].x - points[b].x, points[a].y - points[b].y)


def ear(points, ids):
    horizontal = distance(points, ids[0], ids[3])
    vertical = (distance(points, ids[1], ids[5]) + distance(points, ids[2], ids[4])) / 2
    return vertical / horizontal if horizontal else 0.0


def measure(points):
    eye = (ear(points, LEFT_EYE) + ear(points, RIGHT_EYE)) / 2
    mouth_width = distance(points, 61, 291)
    mouth = distance(points, 13, 14) / mouth_width if mouth_width else 0.0
    return eye, mouth


def head_pose(shape, points):
    height, width = shape[:2]
    image_points = np.array([(points[i].x * width, points[i].y * height) for i in POSE_IDS], dtype=np.float64)
    focal = float(width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    success, rotation, _ = cv2.solvePnP(POSE_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    return tuple(float(v) for v in cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0])


def relative_pitch(samples, rate):
    values = [row["Pitch (deg)"] for row in samples if row["Pitch (deg)"] is not None]
    if not values:
        return
    baseline = float(np.median(values[:min(len(values), max(3, round(rate * 2)))]))
    for row in samples:
        value = row["Pitch (deg)"]
        row["Pitch change (deg)"] = abs((value - baseline + 180) % 360 - 180) if value is not None else None


def find_cues(samples, rate, eye_limit, mouth_limit):
    limits = {"eye": max(2, math.ceil(rate * 1.5) + 1), "mouth": max(2, math.ceil(rate) + 1), "pitch": max(2, math.ceil(rate) + 1)}
    starts, runs, reported, events = {}, {}, {}, []
    for row in samples:
        if not row["Face detected"]:
            starts, runs, reported = {}, {}, {}
            continue
        second = row["Video time (s)"]
        cues = [
            ("eye", row["EAR"] is not None and row["EAR"] < eye_limit, "Sustained low eye opening", f"EAR below {eye_limit:.2f}"),
            ("mouth", row["MAR"] is not None and row["MAR"] > mouth_limit, "Sustained mouth opening", f"MAR above {mouth_limit:.2f}; speech can also cause this"),
            ("pitch", row["Pitch change (deg)"] is not None and row["Pitch change (deg)"] >= 20, "Sustained head-pitch change", "Pitch changed by at least 20° from initial pose"),
        ]
        for key, active, name, evidence in cues:
            if active:
                starts[key] = second if key not in starts else starts[key]
                runs[key] = runs.get(key, 0) + 1
                if runs[key] >= limits[key] and not reported.get(key, False):
                    events.append({"Cue": name, "Started (s)": round(starts[key], 2), "Rule met (s)": round(second, 2), "Evidence": f"{evidence} for about {second - starts[key]:.1f} s"})
                    reported[key] = True
            else:
                starts.pop(key, None)
                runs[key] = 0
                reported[key] = False
    return events


def find_blinks(samples, eye_limit=0.20, maximum_closure_seconds=0.8):
    """Estimate brief eye closures from sampled EAR values; this is not an IR blink sensor."""
    blinks = []
    closed_at = None
    closed_frames = 0
    suppressed_until_open = False
    for row in samples:
        second = row["Video time (s)"]
        eye = row["EAR"]
        if not row["Face detected"] or eye is None:
            closed_at = None
            closed_frames = 0
            suppressed_until_open = False
            continue
        if eye < eye_limit:
            if suppressed_until_open:
                continue
            if closed_at is None:
                closed_at = second
                closed_frames = 0
            closed_frames += 1
            if second - closed_at > maximum_closure_seconds:
                closed_at = None
                closed_frames = 0
                suppressed_until_open = True
            continue
        if closed_at is not None and not suppressed_until_open:
            duration = second - closed_at
            if closed_frames > 0 and 0 < duration <= maximum_closure_seconds:
                blinks.append({"Started (s)": round(closed_at, 2), "Reopened (s)": round(second, 2), "Duration (s)": round(duration, 2)})
        closed_at = None
        closed_frames = 0
        suppressed_until_open = False
    return blinks


def create_landmarker():
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


@lru_cache(maxsize=1)
def _get_live_landmarker():
    return create_landmarker()


def analyze_live_frame(frame_bytes):
    """Measure one JPEG webcam frame using the same landmarks as video analysis."""
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The webcam frame could not be decoded. Please restart the camera.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with _LIVE_DETECT_LOCK:
        result = _get_live_landmarker().detect(image)
    if not result.face_landmarks:
        return {"face_detected": False, "landmarks": []}

    points = result.face_landmarks[0]
    eye, mouth = measure(points)
    angles = head_pose(frame.shape, points)
    xs, ys = [point.x for point in points], [point.y for point in points]
    x0, x1 = max(0, int(min(xs) * frame.shape[1])), min(frame.shape[1], int(max(xs) * frame.shape[1]))
    y0, y1 = max(0, int(min(ys) * frame.shape[0])), min(frame.shape[0], int(max(ys) * frame.shape[0]))
    brightness = None
    if x1 > x0 and y1 > y0:
        brightness = float(np.mean(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)))
    return {
        "face_detected": True,
        "EAR": float(eye),
        "MAR": float(mouth),
        "Pitch (deg)": angles[0] if angles else None,
        "Yaw (deg)": angles[1] if angles else None,
        "Roll (deg)": angles[2] if angles else None,
        "Face brightness": brightness,
        "landmarks": [{"x": float(point.x), "y": float(point.y)} for point in points],
    }


def analyze_video(path, filename, rate=5, eye_limit=0.20, mouth_limit=0.30):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open the video. Try an MP4 encoded with H.264.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and math.isfinite(fps) and fps > 0 else 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / rate))
    measurements, brightnesses = [], []
    sampled = detected = frame_no = 0
    detector = create_landmarker()
    try:
        while sampled < 300:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_no % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = detector.detect(image)
                sampled += 1
                eye = mouth = pitch = yaw = roll = brightness = None
                if result.face_landmarks:
                    detected += 1
                    points = result.face_landmarks[0]
                    eye, mouth = measure(points)
                    angles = head_pose(frame.shape, points)
                    if angles is not None:
                        pitch, yaw, roll = angles
                    xs, ys = [p.x for p in points], [p.y for p in points]
                    x0, x1 = max(0, int(min(xs) * frame.shape[1])), min(frame.shape[1], int(max(xs) * frame.shape[1]))
                    y0, y1 = max(0, int(min(ys) * frame.shape[0])), min(frame.shape[0], int(max(ys) * frame.shape[0]))
                    if x1 > x0 and y1 > y0:
                        brightness = float(np.mean(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)))
                        brightnesses.append(brightness)
                measurements.append({
                    "Sample": sampled, "Video time (s)": frame_no / fps,
                    "EAR": eye, "MAR": mouth, "Pitch (deg)": pitch, "Yaw (deg)": yaw,
                    "Roll (deg)": roll, "Pitch change (deg)": None,
                    "Face detected": eye is not None, "Face brightness": brightness,
                    "Low eye opening": eye is not None and eye < eye_limit,
                    "Open mouth": mouth is not None and mouth > mouth_limit,
                })
            frame_no += 1
    finally:
        detector.close()
        capture.release()

    relative_pitch(measurements, rate)
    coverage = detected / sampled if sampled else 0.0
    average_brightness = float(np.mean(brightnesses)) if brightnesses else 0.0
    adequate = coverage >= 0.70 and average_brightness >= 45
    valid = [row for row in measurements if row["EAR"] is not None]
    events = find_cues(measurements, rate, eye_limit, mouth_limit) if adequate else []
    blinks = find_blinks(measurements, eye_limit) if adequate else []
    observed_duration = max(1 / rate, measurements[-1]["Video time (s)"] - measurements[0]["Video time (s)"]) if measurements else 0
    record = {
        "Video": filename,
        "Analyzed at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Frames": sampled,
        "Face coverage": f"{coverage:.0%}",
        "Face brightness": round(average_brightness, 1) if brightnesses else None,
        "Visual evidence": "Adequate" if adequate else "Insufficient",
        "Average EAR": round(float(np.mean([row["EAR"] for row in valid])), 3) if valid else None,
        "Average MAR": round(float(np.mean([row["MAR"] for row in valid])), 3) if valid else None,
        "Blink count (estimate)": len(blinks),
        "Blink rate (estimate/min)": round(len(blinks) / observed_duration * 60, 1) if observed_duration else 0.0,
        "Sustained cues": len(events),
    }
    return {"analysis": record, "events": events, "measurements": measurements, "frames_total": total_frames}
