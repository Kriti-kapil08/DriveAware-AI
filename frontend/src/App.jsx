import { useEffect, useMemo, useRef, useState } from "react";

async function readApiJson(response) {
  const body = await response.text();
  let data;
  try {
    data = body ? JSON.parse(body) : {};
  } catch {
    throw new Error(`DriveAware API returned an invalid response (HTTP ${response.status}). Check that the Flask server is running.`);
  }
  if (!response.ok) throw new Error(data.error || `DriveAware API request failed (HTTP ${response.status}).`);
  return data;
}

const navItems = [
  { id: "overview", icon: "◫", label: "Overview" },
  { id: "analyze", icon: "◉", label: "Analyze video" },
  { id: "live", icon: "◎", label: "Live monitoring" },
  { id: "history", icon: "◷", label: "Session history" },
  { id: "assistant", icon: "✳", label: "Ask DriveAware" },
];

function getSession() {
  let id = sessionStorage.getItem("driveaware-session-id");
  let started = sessionStorage.getItem("driveaware-session-started");
  if (!id) {
    id = crypto.randomUUID();
    started = new Date().toLocaleString();
    sessionStorage.setItem("driveaware-session-id", id);
    sessionStorage.setItem("driveaware-session-started", started);
  }
  return { id, started };
}

function Metric({ label, value, note, tone = "" }) {
  return <article className={`metric-card ${tone}`}><span className="metric-label">{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

function EarChart({ rows }) {
  const valid = rows.filter((row) => Number.isFinite(row.EAR) && Number.isFinite(row.MAR));
  if (valid.length < 2) return <div className="chart-empty">Measurements will appear here after analysis.</div>;
  const width = 720, height = 210, padX = 12, padY = 18;
  const x = (index) => padX + index * (width - padX * 2) / (valid.length - 1);
  const y = (value) => height - padY - Math.max(0, Math.min(0.8, value)) / 0.8 * (height - padY * 2);
  const pathFor = (key) => valid.map((row, index) => `${index ? "L" : "M"}${x(index)},${y(row[key])}`).join(" ");
  return <div className="chart-wrap"><div className="chart-legend"><span><i className="legend-eye"/> Eye opening (EAR)</span><span><i className="legend-mouth"/> Mouth opening (MAR)</span></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Eye and mouth opening measurements">
    {[0, .2, .4, .6, .8].map((value) => <g key={value}><line x1="0" x2={width} y1={y(value)} y2={y(value)} className="grid-line"/><text x="0" y={y(value) - 4} className="axis-label">{value.toFixed(1)}</text></g>)}
    <path d={pathFor("EAR")} className="line-eye"/><path d={pathFor("MAR")} className="line-mouth"/>
  </svg></div>;
}

function LiveMonitor({ session, speak, onSaved }) {
  const videoRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const measurementsRef = useRef([]);
  const eventsRef = useRef([]);
  const runsRef = useRef({});
  const blinkRunRef = useRef({ closedAt: null, suppressed: false });
  const blinkCountRef = useRef(0);
  const startTimeRef = useRef(0);
  const pitchBaselineRef = useRef(null);
  const alertsRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [saving, setSaving] = useState(false);
  const [voiceAlerts, setVoiceAlerts] = useState(false);
  const [measurement, setMeasurement] = useState(null);
  const [measurements, setMeasurements] = useState([]);
  const [events, setEvents] = useState([]);
  const [blinkCount, setBlinkCount] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [quality, setQuality] = useState({ ready: false, coverage: 0, brightness: null, sampleCount: 0 });
  const [liveError, setLiveError] = useState("");

  useEffect(() => () => {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function startMonitoring() {
    setLiveError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveError("Live camera access requires localhost or a secure HTTPS page in a supported browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      measurementsRef.current = [];
      eventsRef.current = [];
      runsRef.current = {};
      blinkRunRef.current = { closedAt: null, suppressed: false };
      blinkCountRef.current = 0;
      pitchBaselineRef.current = null;
      setMeasurements([]);
      setEvents([]);
      setBlinkCount(0);
      setElapsedSeconds(0);
      setMeasurement(null);
      setQuality({ ready: false, coverage: 0, brightness: null, sampleCount: 0 });
      startTimeRef.current = performance.now();
      runningRef.current = true;
      setRunning(true);
      captureNextFrame();
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setLiveError(error.name === "NotAllowedError" ? "Camera permission was not granted. Allow camera access in the browser, then try again." : `Could not start the camera: ${error.message}`);
    }
  }

  function updateCueRuns(row, elapsed, evidenceAdequate) {
    if (!row["Face detected"] || !evidenceAdequate) {
      runsRef.current = {};
      return;
    }
    const candidates = [
      { key: "eye", active: row.EAR < 0.20, duration: 1.5, cue: "Sustained low eye opening", evidence: "EAR below 0.20" },
      { key: "mouth", active: row.MAR > 0.30, duration: 1.0, cue: "Sustained mouth opening", evidence: "MAR above 0.30; speech can also cause this" },
      { key: "pitch", active: row["Pitch change (deg)"] >= 20, duration: 1.0, cue: "Sustained head-pitch change", evidence: "Pitch changed by at least 20° from the initial pose" },
    ];
    for (const item of candidates) {
      const run = runsRef.current[item.key] || { start: null, reported: false };
      if (!item.active) {
        runsRef.current[item.key] = { start: null, reported: false };
        continue;
      }
      if (run.start === null) run.start = elapsed;
      if (!run.reported && elapsed - run.start >= item.duration) {
        const event = { Cue: item.cue, "Started (s)": Number(run.start.toFixed(2)), "Rule met (s)": Number(elapsed.toFixed(2)), Evidence: `${item.evidence} for about ${(elapsed - run.start).toFixed(1)} s` };
        eventsRef.current = [...eventsRef.current, event];
        setEvents(eventsRef.current);
        run.reported = true;
        if (alertsRef.current) speak(`DriveAware visual cue: ${item.cue}. Please review the situation when safe.`);
      }
      runsRef.current[item.key] = run;
    }
  }

  function updateBlinkEstimate(row, elapsed, evidenceAdequate) {
    const state = blinkRunRef.current;
    if (!row["Face detected"] || !evidenceAdequate) {
      state.closedAt = null;
      state.suppressed = false;
      return;
    }
    if (row.EAR < 0.20) {
      if (state.suppressed) return;
      if (state.closedAt == null) state.closedAt = elapsed;
      if (elapsed - state.closedAt > 0.8) {
        state.closedAt = null;
        state.suppressed = true;
      }
      return;
    }
    if (state.closedAt != null) {
      const duration = elapsed - state.closedAt;
      if (duration >= 0.08 && duration <= 0.8) {
        blinkCountRef.current += 1;
        setBlinkCount(blinkCountRef.current);
      }
    }
    state.closedAt = null;
    state.suppressed = false;
  }

  async function captureNextFrame() {
    if (!runningRef.current) return;
    try {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        window.setTimeout(captureNextFrame, 300);
        return;
      }
      const canvas = captureRef.current;
      canvas.width = Math.min(640, video.videoWidth);
      canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
      if (!blob) throw new Error("Could not capture a camera frame.");
      const form = new FormData();
      form.append("frame", blob, "webcam.jpg");
      const response = await fetch("/api/live-frame", { method: "POST", body: form });
      const result = await readApiJson(response);
      const elapsed = (performance.now() - startTimeRef.current) / 1000;
      if (result["Pitch (deg)"] != null && pitchBaselineRef.current == null) pitchBaselineRef.current = result["Pitch (deg)"];
      const row = {
        Sample: measurementsRef.current.length + 1,
        "Video time (s)": Number(elapsed.toFixed(2)),
        EAR: result.EAR ?? null,
        MAR: result.MAR ?? null,
        "Pitch (deg)": result["Pitch (deg)"] ?? null,
        "Yaw (deg)": result["Yaw (deg)"] ?? null,
        "Roll (deg)": result["Roll (deg)"] ?? null,
        "Pitch change (deg)": result["Pitch (deg)"] != null && pitchBaselineRef.current != null ? Math.abs(result["Pitch (deg)"] - pitchBaselineRef.current) : null,
        "Face detected": Boolean(result.face_detected),
        "Face brightness": result["Face brightness"] ?? null,
        "Low eye opening": Boolean(result.face_detected && result.EAR < 0.20),
        "Open mouth": Boolean(result.face_detected && result.MAR > 0.30),
        landmarks: result.landmarks || [],
      };
      const { landmarks, ...storedRow } = row;
      measurementsRef.current = [...measurementsRef.current, storedRow];
      setMeasurements(measurementsRef.current);
      setMeasurement(row);
      const recent = measurementsRef.current.slice(-10);
      const recentFaces = recent.filter((sample) => sample["Face detected"]);
      const recentBrightness = recentFaces.map((sample) => sample["Face brightness"]).filter((value) => value != null);
      const rollingCoverage = recent.length ? recentFaces.length / recent.length : 0;
      const rollingBrightness = recentBrightness.length ? recentBrightness.reduce((sum, value) => sum + value, 0) / recentBrightness.length : null;
      const evidenceAdequate = recent.length >= 4 && rollingCoverage >= 0.70 && rollingBrightness != null && rollingBrightness >= 45;
      setQuality({ ready: evidenceAdequate, coverage: rollingCoverage, brightness: rollingBrightness, sampleCount: recent.length });
      setElapsedSeconds(elapsed);
      updateBlinkEstimate(row, elapsed, evidenceAdequate);
      updateCueRuns(row, elapsed, evidenceAdequate);
    } catch (error) {
      setLiveError(error.message || "The webcam frame could not be analyzed.");
      runningRef.current = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setRunning(false);
      return;
    }
    if (runningRef.current) window.setTimeout(captureNextFrame, 200);
  }

  async function stopMonitoring() {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setRunning(false);
    if (!measurementsRef.current.length) return;
    const durationSeconds = (performance.now() - startTimeRef.current) / 1000;
    setElapsedSeconds(durationSeconds);
    setSaving(true);
    try {
      const response = await fetch("/api/live-session", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.id, session_started_at: session.started, measurements: measurementsRef.current, events: eventsRef.current, blink_count: blinkCountRef.current, duration_seconds: durationSeconds }),
      });
      const result = await readApiJson(response);
      onSaved({ ...result, measurements: measurementsRef.current });
    } catch (error) {
      setLiveError(`Camera stopped, but the session could not be saved: ${error.message}`);
    } finally { setSaving(false); }
  }

  alertsRef.current = voiceAlerts;
  const latest = measurement;
  const coverage = measurements.length ? Math.round(measurements.filter((row) => row["Face detected"]).length / measurements.length * 100) : 0;
  const blinkRate = elapsedSeconds > 0 ? blinkCount / elapsedSeconds * 60 : 0;
  const qualityTitle = !running ? (measurements.length ? "Monitoring stopped" : "Camera is off") : quality.ready ? "Visual evidence adequate" : "Cue checks paused";
  const qualityMessage = !running
    ? (measurements.length ? "Start a new monitoring session to collect frames and resume cue checks." : "Start monitoring to collect camera frames. Cue checks remain paused while the camera is off.")
    : quality.sampleCount < 4
      ? "Collecting a few frames before checking cues."
      : quality.ready
        ? `Recent face coverage ${Math.round(quality.coverage * 100)}% · brightness ${quality.brightness.toFixed(0)}/255`
        : "Keep the face visible in good lighting; alerts resume when recent coverage is at least 70% and brightness is at least 45/255.";

  return <section className="panel live-panel">
    <div className="section-kicker">LIVE WEBCAM · LOCAL ANALYSIS</div>
    <h2>Live driver monitoring</h2>
    <p className="section-intro">Allow camera access to measure facial cues on this computer. No microphone audio is collected. Keep the face centered and well lit.</p>
    {liveError && <div className="live-error">{liveError}</div>}
    <div className={`quality-banner ${running && quality.ready ? "ready" : "waiting"}`}><b>{qualityTitle}</b><span>{qualityMessage}</span></div>
    <div className="live-video-wrap">
      <video ref={videoRef} autoPlay muted playsInline className="live-video" />
      {latest?.landmarks?.length > 0 && <svg className="landmark-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Face landmark overlay">{latest.landmarks.filter((_, index) => index % 4 === 0).map((point, index) => <circle key={index} cx={point.x} cy={point.y} r="0.003" />)}</svg>}
      {!running && <div className="camera-placeholder"><span>◎</span><b>Camera is off</b><small>Your camera starts only when you press Start monitoring.</small></div>}
      <div className={`live-indicator ${running ? "on" : ""}`}><i />{running ? "LIVE" : "OFF"}</div>
    </div>
    <canvas ref={captureRef} hidden />
    <div className="live-controls">
      {!running ? <button className="primary-button" onClick={startMonitoring} disabled={saving}>{saving ? "Saving session…" : measurements.length ? "Start new monitoring session" : "Start monitoring"}</button> : <button className="stop-button" onClick={stopMonitoring}>Stop and save session</button>}
      <label className="voice-toggle"><input type="checkbox" checked={voiceAlerts} onChange={(event) => setVoiceAlerts(event.target.checked)} /> Voice alerts for sustained cues</label>
      <span className="live-privacy">Camera frames are processed by your local Flask app.</span>
    </div>
    <div className="live-stats">
      <Metric label="FACE TRACKING" value={latest ? (latest["Face detected"] ? "Detected" : "Not found") : "Waiting"} note="Latest camera frame" tone="accent-green" />
      <Metric label="LIVE EVIDENCE" value={!running ? "Stopped" : quality.ready ? "Adequate" : "Paused"} note={!running ? "Start camera to evaluate frames" : quality.brightness == null ? "Waiting for face and lighting" : `${Math.round(quality.coverage * 100)}% coverage · ${quality.brightness.toFixed(0)}/255 brightness`} tone={running && quality.ready ? "accent-blue" : "accent-amber"} />
      <Metric label="EYE OPENING (EAR)" value={latest?.EAR?.toFixed(3) ?? "—"} note="Lower values indicate less eye opening" tone="accent-amber" />
      <Metric label="BLINK COUNT (EST.)" value={blinkCount} note="Brief sampled eye closures" tone="accent-blue" />
      <Metric label="BLINK RATE (EST./MIN)" value={blinkRate.toFixed(1)} note="Based on this monitoring session" tone="accent-teal" />
      <Metric label="SUSTAINED CUES" value={events.length} note="Across this live session" tone="accent-purple" />
    </div>
    <div className="live-event-list"><h3>Live cue events</h3>{events.length ? events.map((event, index) => <div className="cue-row" key={`${event.Cue}-${index}`}><div className="cue-mark">!</div><div><b>{event.Cue}</b><span>{event.Evidence}</span></div><small>{event["Started (s)"]} s</small></div>) : <p>No sustained cues recorded so far.</p>}</div>
    <p className="live-disclaimer">Demonstration only. A measured visual cue is not a fatigue diagnosis or a driving safety decision.</p>
  </section>;
}

export default function App() {
  const session = useMemo(getSession, []);
  const [page, setPage] = useState("overview");
  const [records, setRecords] = useState([]);
  const [active, setActive] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chat, setChat] = useState([]);
  const [question, setQuestion] = useState("");
  const fileRef = useRef(null);

  async function refreshHistory() {
    try {
      const response = await fetch("/api/history");
      const data = await readApiJson(response);
      setRecords(data.records || []);
      const current = (data.records || []).find((record) => record.session_id === session.id);
      if (current && !active) setActive(current);
    } catch (err) { setError(err.message); }
  }

  useEffect(() => { refreshHistory(); }, []);

  const currentRecords = records.filter((record) => record.session_id === session.id);
  const currentEvents = currentRecords.flatMap((record) => record.events || []);
  const latest = active?.analysis || currentRecords[0]?.analysis || null;
  const meanCoverage = currentRecords.length
    ? currentRecords.reduce((sum, record) => sum + Number.parseFloat(record.analysis["Face coverage"] || "0"), 0) / currentRecords.length
    : 0;

  async function analyzeVideo() {
    if (!file) return;
    setLoading(true); setError("");
    const form = new FormData();
    form.append("video", file);
    form.append("session_id", session.id);
    form.append("session_started_at", session.started);
    form.append("sample_rate", "8");
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const result = await readApiJson(response);
      const entry = { session_id: session.id, session_started_at: session.started, analyzed_at: result.analysis["Analyzed at"], analysis: result.analysis, events: result.events, measurements: result.measurements };
      setActive(entry);
      setRecords((old) => [entry, ...old]);
      setPage("overview");
      setFile(null);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function ask(event) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt) return;
    setChat((old) => [...old, { role: "user", text: prompt }]);
    setQuestion("");
    try {
      const response = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt, session_id: session.id }),
      });
      const result = await readApiJson(response);
      setChat((old) => [...old, { role: "assistant", ...result }]);
    } catch (err) {
      setChat((old) => [...old, { role: "assistant", answer: err.message, method: "error" }]);
    }
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-IN";
    window.speechSynthesis.speak(utterance);
  }

  function chooseFile(nextFile) {
    if (!nextFile) return;
    const allowed = [".mp4", ".mov", ".avi", ".m4v"];
    const ext = `.${nextFile.name.split(".").pop().toLowerCase()}`;
    if (!allowed.includes(ext)) { setError("Choose an MP4, MOV, AVI, or M4V video."); return; }
    setError(""); setFile(nextFile);
  }

  function addLiveRecord(entry) {
    setActive(entry);
    setRecords((old) => [entry, ...old.filter((record) => record.analysis._record_id !== entry.analysis._record_id)]);
  }

  function downloadHistoryCsv() {
    const headers = ["Analyzed at", "Session started", "Analysis", "Frames", "Face coverage", "Visual evidence", "Face brightness", "Average EAR", "Average MAR", "Blink count (estimate)", "Blink rate (estimate/min)", "Cue", "Started (s)", "Rule met (s)", "Evidence"];
    const escapeCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const rows = records.flatMap((record) => {
      const events = record.events?.length ? record.events : [null];
      return events.map((event) => [
        record.analyzed_at || record.analysis["Analyzed at"],
        record.session_started_at,
        record.analysis.Video,
        record.analysis.Frames,
        record.analysis["Face coverage"],
        record.analysis["Visual evidence"],
        record.analysis["Face brightness"],
        record.analysis["Average EAR"],
        record.analysis["Average MAR"],
        record.analysis["Blink count (estimate)"],
        record.analysis["Blink rate (estimate/min)"],
        event?.Cue,
        event?.["Started (s)"],
        event?.["Rule met (s)"],
        event?.Evidence,
      ]);
    });
    const csv = [headers, ...rows].map((row) => row.map(escapeCell).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `driveaware-session-history-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">D</div><div><b>DriveAware</b><span>AI MONITORING</span></div></div>
      <div className="side-section-label">WORKSPACE</div>
      <nav>{navItems.map((item) => <button key={item.id} className={`nav-item ${page === item.id ? "selected" : ""}`} onClick={() => setPage(item.id)}><span className="nav-icon">{item.icon}</span>{item.label}</button>)}</nav>
      <div className="side-status"><span className="pulse-dot"/> LOCAL SESSION <b>ACTIVE</b><small>Started {session.started}</small></div>
      <div className="side-disclaimer">Academic demonstration only.<br/>Visual cues are not a fatigue diagnosis.</div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div><span className="eyebrow">DRIVER MONITORING RESEARCH</span><h1>{page === "overview" ? "Monitoring overview" : navItems.find((item) => item.id === page)?.label}</h1></div><div className="topbar-right"><span className="local-pill"><i/> Local processing</span><div className="avatar">K</div></div></header>
      <div className="content">
        <div className="notice"><span>ⓘ</span><p><b>Research demo</b> — DriveAware reports visual cues only. It cannot determine whether a person is safe to drive.</p></div>
        {error && <div className="error-banner"><span>!</span>{error}<button onClick={() => setError("")}>Dismiss</button></div>}

        {page === "overview" && <>
          <div className="welcome-row"><div><h2>Good day, Kriti</h2><p>Here’s the current status of your monitoring session.</p></div><button className="primary-button" onClick={() => setPage("analyze")}>＋ <span>Analyze a video</span></button></div>
          <section className="metric-grid">
            <Metric label="SESSION ANALYSES" value={currentRecords.length} note="Videos analyzed this session" tone="accent-blue" />
            <Metric label="FACE COVERAGE" value={latest?.["Face coverage"] || "—"} note="Latest visual tracking result" tone="accent-green" />
            <Metric label="SUSTAINED CUES" value={currentEvents.length} note="Across the current session" tone={currentEvents.length ? "accent-amber" : "accent-purple"} />
            <Metric label="EVIDENCE QUALITY" value={latest?.["Visual evidence"] || "Waiting"} note={latest ? `Brightness ${latest["Face brightness"] ?? "—"}/255` : "Analyze a clip to begin"} tone="accent-teal" />
          </section>
          <section className="panel analysis-panel"><div className="panel-heading"><div><h3>{active?.analysis?.Video || "Latest analysis"}</h3><p>{latest ? `Analyzed ${active?.analyzed_at || latest["Analyzed at"]}` : "Your analyzed video and measurements will appear here."}</p></div>{latest && <span className={`status-badge ${latest["Visual evidence"] === "Adequate" ? "good" : "warn"}`}>{latest["Visual evidence"] === "Adequate" ? "●" : "●"} {latest["Visual evidence"]}</span>}</div>
            {latest ? <><div className="detail-metrics"><div><small>AVERAGE EAR</small><b>{latest["Average EAR"] ?? "—"}</b></div><div><small>AVERAGE MAR</small><b>{latest["Average MAR"] ?? "—"}</b></div><div><small>FRAMES</small><b>{latest.Frames}</b></div><div><small>FACE BRIGHTNESS</small><b>{latest["Face brightness"] ?? "—"}<em>/255</em></b></div><div><small>BLINK COUNT (EST.)</small><b>{latest["Blink count (estimate)"] ?? "—"}</b></div><div><small>BLINK RATE (EST./MIN)</small><b>{latest["Blink rate (estimate/min)"] ?? "—"}</b></div></div><EarChart rows={active?.measurements || []}/></> : <div className="empty-state"><div className="empty-icon">◉</div><b>No analysis yet</b><span>Upload a short clip to see the visual measurements.</span><button className="secondary-button" onClick={() => setPage("analyze")}>Open video analysis</button></div>}
          </section>
          <section className="panel"><div className="panel-heading"><div><h3>Recent cues</h3><p>Sustained measurements with their rule evidence</p></div><button className="text-button" onClick={() => setPage("history")}>View saved history →</button></div>
            {currentEvents.length ? <div className="cue-list">{currentEvents.slice(0, 4).map((event, index) => <div className="cue-row" key={`${event.Cue}-${index}`}><div className="cue-mark">!</div><div><b>{event.Cue}</b><span>{event.Evidence}</span></div><small>{event["Started (s)"]} s</small></div>)}</div> : <div className="quiet-state">No sustained cues recorded in this session.</div>}
          </section>
        </>}

        {page === "analyze" && <section className="panel upload-page"><div className="section-kicker">VIDEO ANALYSIS</div><h2>Analyze a recorded clip</h2><p className="section-intro">Upload a short driver-facing video. DriveAware samples frames, measures facial geometry, and reviews sustained visual cues.</p>
          <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files[0]); }} onClick={() => fileRef.current?.click()}><input ref={fileRef} type="file" accept="video/mp4,video/quicktime,video/x-msvideo,.m4v" hidden onChange={(e) => chooseFile(e.target.files[0])}/><div className="upload-icon">↑</div><b>{file ? file.name : "Drop your video here, or browse"}</b><span>MP4, MOV, AVI, M4V · Up to 200 MB</span><button className="secondary-button" type="button">Choose video</button></div>
          <div className="privacy-note"><span>⌂</span><div><b>Processed locally</b><p>Your clip is analyzed on this computer and is not uploaded to a third-party service.</p></div></div>
          <div className="form-actions"><span>Analysis uses the current demonstration thresholds.</span><button className="primary-button" disabled={!file || loading} onClick={analyzeVideo}>{loading ? <><span className="spinner"/> Analyzing…</> : "Analyze clip →"}</button></div>
          {active?.analysis && <div className="inline-result"><b>Most recent result</b><span>{active.analysis["Face coverage"]} face coverage · {active.analysis["Sustained cues"]} sustained cue(s)</span></div>}
        </section>}

        {page === "live" && <LiveMonitor session={session} speak={speak} onSaved={addLiveRecord} />}

        {page === "history" && <section className="panel history-panel"><div className="panel-heading"><div><h3>Saved analysis history</h3><p>Completed webcam and recorded-video analyses saved locally</p></div><div className="history-heading-actions"><span className="count-pill">{records.length} records</span><button className="secondary-button export-button" onClick={downloadHistoryCsv} disabled={!records.length}>Download CSV</button></div></div>
          {records.length ? <div className="table-scroll"><table><thead><tr><th>SESSION</th><th>ANALYZED</th><th>FRAMES</th><th>FACE COVERAGE</th><th>EVIDENCE</th><th>CUES</th><th/></tr></thead><tbody>{records.map((record, index) => <tr key={record.analysis._record_id || `${record.analyzed_at}-${index}`}><td><b>{record.analysis.Video}</b></td><td>{record.analyzed_at}</td><td>{record.analysis.Frames}</td><td>{record.analysis["Face coverage"]}</td><td><span className={`status-badge ${record.analysis["Visual evidence"] === "Adequate" ? "good" : "warn"}`}>{record.analysis["Visual evidence"]}</span></td><td>{record.events.length}</td><td className="history-actions"><button className="text-button" onClick={() => setSelectedRecord(record)}>Details</button><button className="icon-button" title="Open overview" onClick={() => {setActive(record); setPage("overview");}}>↗</button></td></tr>)}</tbody></table></div> : <div className="empty-state compact"><div className="empty-icon">◷</div><b>No saved analyses</b><span>Completed analyses will appear here.</span></div>}
          {selectedRecord && <div className="history-detail"><div className="history-detail-heading"><div><span className="section-kicker">ANALYSIS DETAILS</span><h3>{selectedRecord.analysis.Video}</h3><p>{selectedRecord.session_started_at || "Session"} · Analyzed {selectedRecord.analyzed_at}</p></div><button className="icon-button" aria-label="Close analysis details" onClick={() => setSelectedRecord(null)}>×</button></div>
            <div className="history-summary"><div><small>FACE COVERAGE</small><b>{selectedRecord.analysis["Face coverage"]}</b></div><div><small>EVIDENCE QUALITY</small><b>{selectedRecord.analysis["Visual evidence"]}</b></div><div><small>AVERAGE EAR</small><b>{selectedRecord.analysis["Average EAR"] ?? "—"}</b></div><div><small>AVERAGE MAR</small><b>{selectedRecord.analysis["Average MAR"] ?? "—"}</b></div><div><small>FACE BRIGHTNESS</small><b>{selectedRecord.analysis["Face brightness"] ?? "—"}/255</b></div><div><small>SAMPLED FRAMES</small><b>{selectedRecord.analysis.Frames}</b></div><div><small>BLINK COUNT (EST.)</small><b>{selectedRecord.analysis["Blink count (estimate)"] ?? "—"}</b></div><div><small>BLINK RATE (EST./MIN)</small><b>{selectedRecord.analysis["Blink rate (estimate/min)"] ?? "—"}</b></div></div>
            {selectedRecord.measurements?.length > 1 && <div className="history-chart"><h4>Measurements across sampled frames</h4><EarChart rows={selectedRecord.measurements} /></div>}
            <h4>Recorded cue evidence</h4>
            {selectedRecord.events?.length ? <div className="table-scroll"><table><thead><tr><th>CUE</th><th>STARTED (S)</th><th>RULE MET (S)</th><th>EVIDENCE</th></tr></thead><tbody>{selectedRecord.events.map((event, index) => <tr key={`${event.Cue}-${index}`}><td><b>{event.Cue}</b></td><td>{event["Started (s)"]}</td><td>{event["Rule met (s)"]}</td><td>{event.Evidence}</td></tr>)}</tbody></table></div> : <p className="history-no-events">No sustained cues were recorded in this analysis.</p>}
          </div>}
        </section>}

        {page === "assistant" && <section className="panel assistant-panel"><div className="section-kicker">SESSION ASSISTANT</div><h2>Ask DriveAware</h2><p className="section-intro">Answers use the current session’s saved visual events and evidence checks.</p><div className="chat-window">{chat.length === 0 && <div className="empty-state compact"><div className="empty-icon">✳</div><b>Ask about your session</b><span>Try “Why was a cue recorded?” or “Summarize this session.”</span></div>}{chat.map((message, index) => <div className={`chat-message ${message.role}`} key={index}><div className="chat-avatar">{message.role === "user" ? "K" : "D"}</div><div className="chat-body"><p>{message.role === "user" ? message.text : message.answer}</p>{message.role === "assistant" && <div className="chat-meta"><span>{message.intent?.replaceAll("_", " ")}</span><span>{message.method}</span>{message.confidence != null && <span>confidence {message.confidence.toFixed(2)}</span>}<button onClick={() => speak(message.answer)}>🔊 Read aloud</button></div>}</div></div>)}</div><form className="chat-form" onSubmit={ask}><input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about recorded cues, evidence, or this session…"/><button className="primary-button" disabled={!question.trim()}>Send ↑</button></form><small className="muted">The assistant reports visual observations, not a diagnosis.</small></section>}

        <footer>DRIVEAWARE AI <span>·</span> Academic research demonstrator <span>·</span> Not a driving safety system</footer>
      </div>
    </main>
  </div>;
}
