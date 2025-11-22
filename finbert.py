import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

CSV_PATH = "all-data.csv" 
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "finbert_model"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LENGTH = 128

df = pd.read_csv(CSV_PATH, encoding="latin1")

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must have 'text' and 'label' columns")

df = df.dropna(subset=["text", "label"])

df["label"] = df["label"].astype(str).str.strip().str.lower()

label_map = {"positive": 2, "neutral": 1, "negative": 0}
df["label"] = df["label"].map(label_map)

num_dropped = df["label"].isna().sum()
if num_dropped > 0:
    print(f"Dropping {num_dropped} rows with invalid labels")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("Dataset ready:", df.shape)
print("Label distribution:\n", df["label"].value_counts())

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df["label"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuned FinBERT model saved to", OUTPUT_DIR)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

