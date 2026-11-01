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

## NLP assistant and intent model

The app includes a small question-intent matcher for why a cue was recorded,
recent events, session summary, camera/visual-evidence status, and help. Answers
are generated from the current Streamlit session's structured clip/event records;
the assistant does not invent missing events. This is a rule-based NLP baseline,
not a fine-tuned transformer.

`data/intent_examples.jsonl` contains six question intents with separate
train/validation/test labels. `train_intent.py` fine-tunes DistilBERT and reports
held-out accuracy, macro-F1, and a confusion matrix for both DistilBERT and the
keyword baseline. The examples are synthetic student-created prompts, so the
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
The keyword baseline remains the default. The trained model scored 0.47 accuracy
and 0.44 macro-F1 on this small held-out synthetic test set, so its output is
experimental; a confidence below 0.55 falls back to the keyword baseline.
Answers are still generated only from structured session records. Keep model
weights out of GitHub; `.gitignore` excludes the checkpoint folder.

Hugging Face documents DistilBERT sequence classification and PyTorch support;
this project uses a small PyTorch training loop. [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) · [Trainer](https://huggingface.co/docs/transformers/en/trainer)
