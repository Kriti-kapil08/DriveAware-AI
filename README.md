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

## New React + Flask dashboard (migration in progress)

The existing Streamlit app remains available while the new interface is built.
The React/Vite dashboard currently connects to a Flask API for recorded-video
analysis, saved analysis history, and grounded session Q&A. Live webcam mode will
be moved in a later migration step.

Install the Flask API dependency in the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

Open two PowerShell terminals in this folder. In the first, start the Flask API:

```powershell
.venv\Scripts\python.exe -m backend.app
```

In the second, install the JavaScript packages and start the React dashboard:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The React
development server forwards `/api` requests to Flask on `http://127.0.0.1:5001`.

## Current scope

The app accepts a short recorded video and includes an experimental live webcam
mode. Both paths use MediaPipe face landmarks to estimate eye opening (EAR), mouth
opening (MAR), and head pose. Sustained cues are reported using consecutive
sampled-frame rules. The live path samples the browser camera at up to five frames
per second, shows a live landmark overlay, and updates face coverage, brightness,
and evidence status over a rolling window. The dashboard header summarizes camera
status, latest evidence, analyses, and sustained cues for the current session.
Inadequate evidence withholds new
live cue events. The app keeps session summaries and cue events in Streamlit
session state and can download summaries as CSV. Current-session chat context
clears when the app session resets. Completed analysis summaries and cue events
are saved locally in SQLite and shown under Saved analysis history; the database
is excluded from Git. These visual cues are not a fatigue diagnosis or safety
system.

The first run downloads the MediaPipe face landmark model from Google's MediaPipe
model storage. Internet access is needed for that download. Live webcam access
requires permission in the browser. The included WebRTC path is intended for
local demonstration; remote webcam access requires HTTPS and suitable WebRTC
network configuration. An optional short Windows notification tone can play
when a sustained cue is recorded; it is off by default and runs on the computer
hosting the local app. It is a demonstration alert, not a safety warning system.
When a sustained cue is recorded, the video and live-monitor results also show a
**Speak sustained-cue alert** button. It speaks the cue name and evidence caution
through the browser's speech engine after the user clicks it.

## NLP assistant and intent model

The app includes a small question-intent matcher for why a cue was recorded,
recent events, session summary, camera/visual-evidence status, and help. Answers
are generated from the current Streamlit session's structured clip/event records;
the assistant does not invent missing events. The keyword matcher is the baseline;
an optional fine-tuned DistilBERT classifier predicts the question intent. Each
assistant answer has a browser-based **Read answer aloud** control using the
browser's built-in English speech voices. Speech begins only when the user clicks
the control.

`data/intent_examples.jsonl` contains six question intents with separate
train/validation/test labels. `train_intent.py` fine-tunes DistilBERT and reports
held-out accuracy, macro-F1, and a confusion matrix for both DistilBERT and the
keyword baseline. It selects a confidence fallback threshold on validation data
by maximizing coverage while meeting a target selective accuracy and a 0.35
minimum threshold, then reports test coverage and selective accuracy at that
threshold. The test set is not used to choose the threshold. The examples are
synthetic student-created prompts, so the
metrics are a small learning baseline, not proof of generalization to all users.
The current data has 23 training, 5 validation, and 5 held-out test questions per
intent; the test questions were kept unchanged while adding more varied training
examples.

For a simple Google Colab run:

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com/).
2. Upload `train_intent.py` and `data/intent_examples.jsonl` using Colab's Files
   panel (upload the JSONL as `intent_examples.jsonl`). A GPU runtime is optional.
3. Run:

   ```python
   !pip install -q transformers scikit-learn
   !python train_intent.py --data intent_examples.jsonl
   ```

4. Zip and download the trained model folder:

   ```python
   import shutil
   from google.colab import files

   shutil.make_archive("driveaware_intent_model", "zip", "driveaware_intent_model")
   files.download("driveaware_intent_model.zip")
   ```

The app can use the local model experimentally. Keep the extracted
`driveaware_intent_model/` folder next to `app.py`, then install the optional
inference dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-nlp.txt
```

In the app, enable **Use experimental DistilBERT intent classifier** to try it.
The keyword baseline remains the default. The previous checkpoint scored 0.87
accuracy and 0.86 macro-F1 on this small held-out synthetic test set. This is a
promising learning result, not evidence of general-user accuracy. The app uses
the validation-selected threshold in `evaluation.json`; older checkpoints
without this field keep the 0.55 default.
Answers are still generated only from structured session records. Keep model
weights out of GitHub; `.gitignore` excludes the checkpoint folder.

Hugging Face documents DistilBERT sequence classification and PyTorch support;
this project uses a small PyTorch training loop. [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) · [Trainer](https://huggingface.co/docs/transformers/en/trainer)
