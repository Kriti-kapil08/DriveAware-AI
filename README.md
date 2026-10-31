# DriveAware AI

A beginner-friendly, two-day prototype for estimating fatigue-related visual cues.

## Run locally on Windows

Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks environment activation, use this command for installing and
running instead:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Current scope

Phase 2 accepts a short recorded video, samples frames, overlays face landmarks,
reports approximate eye (EAR), mouth (MAR), and head-pose measurements, lists
sustained cues using simple consecutive-frame rules, and reports face tracking
coverage and average face-region brightness. It keeps a session summary and cue
event history in Streamlit session state and can download the summary as CSV.
The history is temporary and clears when the app session resets. Head-pitch change is measured
relative to the first two seconds of the clip. Sustained-cue results are withheld
when coverage is below 70% or average face brightness is below 45/255. On first
analysis, the app downloads the MediaPipe face landmark model from Google's
MediaPipe model storage. Internet access is needed for that first download.
These visual cues are not a fatigue diagnosis or safety system.
