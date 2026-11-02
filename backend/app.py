"""Flask JSON API for the React DriveAware dashboard."""

from datetime import datetime
import json
import re
import sqlite3
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from backend.vision import analyze_live_frame, analyze_video

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "driveaware_history.sqlite3"
MODEL_DIR = ROOT / "driveaware_intent_model"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


def connect_db():
    return sqlite3.connect(DB_PATH)


def initialize_db():
    with connect_db() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS saved_analyses (
            record_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            session_started_at TEXT NOT NULL,
            analyzed_at TEXT NOT NULL,
            record_json TEXT NOT NULL,
            events_json TEXT NOT NULL
        )""")


def get_history(session_id=None, limit=100):
    query = "SELECT session_id, session_started_at, analyzed_at, record_json, events_json FROM saved_analyses"
    args = []
    if session_id:
        query += " WHERE session_id = ?"
        args.append(session_id)
    query += " ORDER BY analyzed_at DESC LIMIT ?"
    args.append(limit)
    with connect_db() as connection:
        rows = connection.execute(query, args).fetchall()
    return [{
        "session_id": row[0], "session_started_at": row[1], "analyzed_at": row[2],
        "analysis": json.loads(row[3]), "events": json.loads(row[4]),
    } for row in rows]


def save_analysis(session_id, session_started_at, result):
    record = result["analysis"]
    record_id = str(uuid.uuid4())
    record["_record_id"] = record_id
    with connect_db() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO saved_analyses VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, session_id, session_started_at, record["Analyzed at"],
             json.dumps(record), json.dumps(result["events"])),
        )


def classify_keyword(text):
    tokens = set(re.findall(r"[a-z0-9]+", text.lower().strip()))
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


@lru_cache(maxsize=1)
def load_intent_model():
    if not (MODEL_DIR / "config.json").exists():
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.to(torch.device("cpu"))
    model.eval()
    report_path = MODEL_DIR / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    threshold = float(report.get("confidence_policy", {}).get("threshold", 0.55))
    return torch, tokenizer, model, threshold


def classify(text):
    try:
        bundle = load_intent_model()
    except Exception:
        bundle = None
    if bundle is None:
        return classify_keyword(text), "keyword baseline", None
    torch, tokenizer, model, threshold = bundle
    try:
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
        confidence, label_index = torch.max(probabilities, dim=-1)
        label_index = int(label_index.item())
        id2label = model.config.id2label
        predicted = id2label.get(label_index, id2label.get(str(label_index), "unsupported"))
        confidence = float(confidence.item())
        if confidence < threshold:
            return classify_keyword(text), "keyword baseline (low model confidence)", confidence
        return predicted, "experimental DistilBERT", confidence
    except Exception:
        # The assistant remains usable if the optional local model cannot infer.
        app.logger.exception("Intent model inference failed; using keyword baseline")
        return classify_keyword(text), "keyword baseline (model unavailable)", None


def answer_from_history(intent, records):
    records = list(reversed(records))
    if not records:
        return "No analysis is saved for this session yet. Analyze a video first."
    events = [event for item in records for event in item["events"]]
    latest = records[-1]["analysis"]
    if intent == "why_cue":
        if not events:
            return "No sustained cue was recorded. The app only reports cues that meet their time rule when visual evidence is adequate."
        event = events[-1]
        return (f"In {latest['Video']}, the system recorded {event['Cue']}. It began at about "
                f"{event['Started (s)']} seconds and met the rule at {event['Rule met (s)']} seconds because "
                f"{event['Evidence']}. This is a measured visual cue, not a fatigue diagnosis.")
    if intent == "recent_events":
        if not events:
            return "There are no sustained cues in this session. Short-lived measurements may not meet a cue rule."
        return "Recent recorded cues:\n\n" + "\n".join(
            f"- {event['Cue']} in {item['analysis']['Video']} at {event['Started (s)']} s: {event['Evidence']}"
            for item in records for event in item["events"]
        )
    if intent == "session_summary":
        coverage = sum(float(item["analysis"]["Face coverage"].rstrip("%")) for item in records) / len(records)
        return (f"This session contains {len(records)} analyzed clip(s) and {len(events)} sustained cue(s). "
                f"Average face-tracking coverage was {coverage:.0f}%. These are visual observations, not a diagnosis.")
    if intent == "camera_status":
        return (f"For {latest['Video']}, face-tracking coverage was {latest['Face coverage']}, brightness was "
                f"{latest['Face brightness']}/255, and visual evidence was {latest['Visual evidence']}. "
                "These are simple quality checks, not a complete camera-health test.")
    if intent == "help":
        return "Ask why a cue was recorded, what recent events were detected, for a session summary, or about visual evidence."
    return "I can answer about recorded cues, recent events, session summaries, and visual evidence. Try rephrasing with one of those topics."


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "DriveAware Flask API"})


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a video file first."}), 400
    filename = secure_filename(upload.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Upload an MP4, MOV, AVI, or M4V video."}), 415
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    session_started_at = request.form.get("session_started_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            upload.save(temp)
            video_path = Path(temp.name)
        result = analyze_video(
            video_path, filename,
            rate=max(1, min(8, int(request.form.get("sample_rate", 5)))),
            eye_limit=float(request.form.get("eye_threshold", 0.20)),
            mouth_limit=float(request.form.get("mouth_threshold", 0.30)),
        )
        save_analysis(session_id, session_started_at, result)
        return jsonify({"session_id": session_id, **result})
    except (ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Video analysis failed")
        return jsonify({"error": f"Video analysis failed: {error}"}), 500
    finally:
        if "video_path" in locals():
            video_path.unlink(missing_ok=True)


@app.get("/api/history")
def history():
    return jsonify({"records": get_history(request.args.get("session_id"))})


@app.post("/api/live-frame")
def live_frame():
    frame = request.files.get("frame")
    if frame is None:
        return jsonify({"error": "No webcam frame was received."}), 400
    try:
        return jsonify(analyze_live_frame(frame.read()))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Live webcam frame analysis failed")
        return jsonify({"error": f"Live frame analysis failed: {error}"}), 500


@app.post("/api/live-session")
def save_live_session():
    payload = request.get_json(silent=True) or {}
    measurements = payload.get("measurements") or []
    if not measurements:
        return jsonify({"error": "Start the webcam and collect frames before saving."}), 400
    detected = [row for row in measurements if row.get("Face detected")]
    brightnesses = [row["Face brightness"] for row in detected if row.get("Face brightness") is not None]
    ears = [row["EAR"] for row in detected if row.get("EAR") is not None]
    mars = [row["MAR"] for row in detected if row.get("MAR") is not None]
    coverage = len(detected) / len(measurements)
    brightness = sum(brightnesses) / len(brightnesses) if brightnesses else 0.0
    adequate = coverage >= 0.70 and brightness >= 45
    events = payload.get("events") or []
    record = {
        "Video": "Live webcam session",
        "Analyzed at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Frames": len(measurements),
        "Face coverage": f"{coverage:.0%}",
        "Face brightness": round(brightness, 1) if brightnesses else None,
        "Visual evidence": "Adequate" if adequate else "Insufficient",
        "Average EAR": round(sum(ears) / len(ears), 3) if ears else None,
        "Average MAR": round(sum(mars) / len(mars), 3) if mars else None,
        "Sustained cues": len(events),
    }
    result = {"analysis": record, "events": events}
    session_id = str(payload.get("session_id") or uuid.uuid4())
    session_started_at = str(payload.get("session_started_at") or record["Analyzed at"])
    save_analysis(session_id, session_started_at, result)
    return jsonify({"session_id": session_id, "session_started_at": session_started_at,
                    "analyzed_at": record["Analyzed at"], **result})


@app.post("/api/ask")
def ask():
    try:
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("question", "")).strip()
        if not text:
            return jsonify({"error": "Enter a question."}), 400
        intent, method, confidence = classify(text)
        records = get_history(payload.get("session_id"))
        answer = answer_from_history(intent, records)
        return jsonify({"answer": answer, "intent": intent, "method": method, "confidence": confidence})
    except Exception:
        app.logger.exception("Assistant request failed")
        return jsonify({"error": "The assistant could not read the saved session data. Check the Flask terminal for details."}), 500


initialize_db()

if __name__ == "__main__":
    # Torch and other packages write/update files under site-packages while
    # importing. Flask's debug reloader watches those files and can restart
    # the server during an NLP request, resetting the Vite proxy connection.
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
