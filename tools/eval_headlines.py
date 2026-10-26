"""
Measure the model on REAL headlines you labeled yourself (the number that actually matters).

Step 1 - export unlabeled headlines the pipeline already stored:
    python tools/eval_headlines.py --export 200 --out eval/headlines.csv
Step 2 - open eval/headlines.csv and fill the `label` column with negative / neutral / positive
         (judge the headline's likely effect on the company's stock; ignore the model_* columns).
Step 3 - score it:
    python tools/eval_headlines.py --eval eval/headlines.csv

Rule of thumb: >= 85% accuracy and >= 0.80 macro-F1 on 150+ real headlines means no fine-tuning is
needed. Below ~80%, fine-tune (see README "Fine-tuning").
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = ["negative", "neutral", "positive"]


def export(db: str, n: int, out: str, symbol: str) -> None:
    from streaming.store import Store
    rows = Store(db).headlines(n, symbol)
    if not rows:
        raise SystemExit("Store is empty. Run the stack so headlines accumulate first.")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "title", "label", "model_label", "model_score"])
        for r in rows:
            w.writerow([r["symbol"], r["title"], "", r["label"], f"{r['score']:.3f}"])
    print(f"wrote {len(rows)} headlines to {out}; fill the `label` column, then run --eval")


def evaluate(path: str) -> None:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sentiment_service import get_finbert

    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("label") or "").strip().lower() in LABELS]
    if len(rows) < 30:
        raise SystemExit(f"Only {len(rows)} labeled rows; label at least 30 (150+ recommended).")
    truth = [r["label"].strip().lower() for r in rows]
    preds = []
    scorer = get_finbert()
    for i in range(0, len(rows), 32):
        preds += [x["label"] for x in scorer.predict_batch([r["title"] for r in rows[i:i + 32]])]
    print(f"n={len(rows)}  accuracy={accuracy_score(truth, preds):.3f}  "
          f"macro_f1={f1_score(truth, preds, average='macro', labels=LABELS, zero_division=0):.3f}")
    print(classification_report(truth, preds, labels=LABELS, zero_division=0))
    print("confusion (rows=truth, cols=pred):", LABELS)
    print(confusion_matrix(truth, preds, labels=LABELS))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.getenv("MARKETPULSE_DB", "data/marketpulse.db"))
    p.add_argument("--export", type=int, metavar="N")
    p.add_argument("--symbol", default="")
    p.add_argument("--out", default="eval/headlines.csv")
    p.add_argument("--eval", metavar="CSV")
    a = p.parse_args()
    if a.export:
        export(a.db, a.export, a.out, a.symbol.upper())
    elif a.eval:
        evaluate(a.eval)
    else:
        p.error("use --export N or --eval CSV")


if __name__ == "__main__":
    main()