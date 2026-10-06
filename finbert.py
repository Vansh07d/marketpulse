import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ---------------- Config ----------------
SEED = 42
DATA_PATH = "all-data.csv"
MODEL_NAME = "ProsusAI/finbert"   # try "microsoft/deberta-v3-base" for a stronger backbone
OUTPUT_DIR = "models/finbert_model"
METRICS_PATH = "models/training_metrics.json"

LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE = 2              # epochs without val macro-F1 improvement before stopping
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True  # set False to compare; may help raw accuracy
N_FOLDS = 2
TEST_SIZE = 0.10          # held-out set, never used for training or model selection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Data ----------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)  # padded per batch
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")[["text", "label"]]
    df["text"] = df["text"].astype(str).str.strip()
    df["label_id"] = df["label"].astype(str).str.lower().str.strip().map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df = df[df["text"].str.len() > 0]
    n = len(df)
    df = df.drop_duplicates("text")  # duplicates would leak across splits
    print(f"Dropped {n - len(df)} duplicate sentences -> {len(df)} samples")
    df["label_id"] = df["label_id"].astype(int)
    print(df["label"].value_counts().sort_index())
    return df


# ---------------- Evaluation ----------------
@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true, probs):
    preds = probs.argmax(1)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, preds, labels=[0, 1, 2], target_names=LABELS,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
    }


# ---------------- Training (one fold) ----------------
def train_fold(fold, X_tr, y_tr, X_va, y_va, tokenizer):
    set_seed(SEED + fold)
    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(SentimentDataset(X_tr, y_tr, tokenizer), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SentimentDataset(X_va, y_va, tokenizer), batch_size=BATCH_SIZE,
                            collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS)).to(DEVICE)

    weight = None
    if USE_CLASS_WEIGHTS:
        w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr)
        weight = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    path = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    best_f1, stale, history = -1.0, 0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(**batch).logits
            loss = criterion(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val = metrics(y_va, get_probs(model, val_loader))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(f"[Fold {fold}] Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"| val_acc={val['accuracy']:.4f} | val_macro_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            model.save_pretrained(path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[Fold {fold}] Early stopping.")
                break

    del model, optimizer
    torch.cuda.empty_cache()
    return path, best_f1, history


# ---------------- Main ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE} | bf16: {USE_BF16}")

    df = load_data()
    X, y = df["text"].values, df["label_id"].values
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Dev: {len(X_dev)} | Test: {len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    test_loader = DataLoader(SentimentDataset(X_test, y_test, tokenizer), batch_size=BATCH_SIZE,
                             collate_fn=DataCollatorWithPadding(tokenizer))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    test_probs, folds = [], []

    for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev), start=1):
        path, best_f1, history = train_fold(fold, X_dev[tr], y_dev[tr], X_dev[va], y_dev[va], tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        probs = get_probs(model, test_loader)
        del model
        test_probs.append(probs)

        m = metrics(y_test, probs)
        print(f"[Fold {fold}] best_val_macro_f1={best_f1:.4f} | test_acc={m['accuracy']:.4f} "
              f"| test_macro_f1={m['macro_f1']:.4f}\n")
        folds.append({"fold": fold, "best_val_macro_f1": best_f1, "history": history,
                      "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"]})

    # Ensemble: average softmax probabilities across folds
    ens = metrics(y_test, np.mean(test_probs, axis=0))
    accs = [f["test_accuracy"] for f in folds]
    print("=" * 50)
    print(f"Single-model test acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"ENSEMBLE test acc:     {ens['accuracy']:.4f}")
    print(f"ENSEMBLE weighted F1:  {ens['f1']:.4f} | macro F1: {ens['macro_f1']:.4f}")
    print("Confusion matrix:\n", np.array(ens["confusion_matrix"]))

    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (str, int, float, bool, list, dict))}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": config, "dev_size": len(X_dev), "test_size": len(X_test),
                   "folds": folds, "ensemble_test_metrics": ens}, f, indent=2, default=float)

    print(f"Fold models saved under {OUTPUT_DIR}/fold_* | Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()