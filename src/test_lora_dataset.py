import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "distilbert-base-multilingual-cased"
BASE_DIR = Path(__file__).resolve().parent.parent
LORA_PATH = BASE_DIR / "models" / "transformer_lora"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"

id2label = {0: "moroccan", 1: "oujdi"}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model.eval()

df = pd.read_csv(DATASET_PATH)

samples = pd.concat([
    df[df["label"] == "oujdi"].head(10),
    df[df["label"] == "moroccan"].head(10)
])

for _, row in samples.iterrows():
    text = str(row["text"])
    true_label = row["label"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    pred_label = id2label[pred_id]

    print("TEXT:", text)
    print("TRUE:", true_label, "| PRED:", pred_label, "| PROBS:", probs.tolist())
    print("-" * 80)
