"""Fine-tune DistilBERT for DriveAware question-intent classification.

Use in Google Colab (GPU optional):
    pip install transformers scikit-learn
    python train_intent.py --data intent_examples.jsonl
"""

import argparse
import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "distilbert/distilbert-base-uncased"
EXPECTED_SPLITS = {"train", "validation", "test"}


class IntentDataset(Dataset):
    def __init__(self, rows, tokenizer, label_to_id, max_length=64):
        encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor([label_to_id[row["label"]] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def load_data(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if set(row) != {"text", "label", "split"} or row["split"] not in EXPECTED_SPLITS:
            raise ValueError("Each JSONL row must have text, label, and split=train|validation|test.")
    labels = sorted({row["label"] for row in rows})
    splits = {name: [row for row in rows if row["split"] == name] for name in EXPECTED_SPLITS}
    if any(not splits[name] for name in EXPECTED_SPLITS):
        raise ValueError("The dataset must contain train, validation, and test rows.")
    for split_name, split_rows in splits.items():
        missing = set(labels) - {row["label"] for row in split_rows}
        if missing:
            raise ValueError(f"Split '{split_name}' is missing labels: {sorted(missing)}")
    return splits, labels


def evaluate(model, loader, device):
    model.eval()
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy().tolist()
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            predictions = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
            true_labels.extend(labels)
            predicted_labels.extend(predictions)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
    }


def classify_keyword_baseline(text):
    """Keep this rule order aligned with app.py for a fair comparison."""
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


def evaluate_keyword_baseline(rows, labels):
    label_to_id = {label: index for index, label in enumerate(labels)}
    true_labels = [label_to_id[row["label"]] for row in rows]
    predicted_labels = [label_to_id[classify_keyword_baseline(row["text"])] for row in rows]
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="intent_examples.jsonl")
    parser.add_argument("--output_dir", default="driveaware_intent_model")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    splits, labels = load_data(args.data)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    datasets = {
        name: IntentDataset(rows, tokenizer, label_to_id)
        for name, rows in splits.items()
    }
    train_loader = DataLoader(datasets["train"], batch_size=16, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=32)
    test_loader = DataLoader(datasets["test"], batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_validation_f1 = -1.0
    best_state = None

    print(f"Device: {device}; labels: {labels}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach().cpu())

        validation_metrics = evaluate(model, validation_loader, device)
        mean_loss = total_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={mean_loss:.4f} | "
            f"validation accuracy={validation_metrics['accuracy']:.3f} | "
            f"validation macro-F1={validation_metrics['macro_f1']:.3f}"
        )
        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    distilbert_test_metrics = evaluate(model, test_loader, device)
    keyword_test_metrics = evaluate_keyword_baseline(splits["test"], labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    report = {
        "base_model": MODEL_NAME,
        "labels": labels,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "best_validation_macro_f1": best_validation_f1,
        "test": {
            "distilbert": distilbert_test_metrics,
            "keyword_baseline": keyword_test_metrics,
        },
        "note": "Small synthetic intent dataset; test scores are a learning baseline, not evidence of broad user generalization.",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Held-out test metrics:")
    print(json.dumps(report["test"], indent=2))
    print(f"Saved fine-tuned model and evaluation report to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
