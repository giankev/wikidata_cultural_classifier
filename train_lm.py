# === Imports ===
import os
import json
import pickle
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    set_seed,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import torch
from torch.nn import CrossEntropyLoss
from collections import Counter
from evaluate import load as load_metric

# === Config ===
set_seed(42)
LABELS = ["cultural agnostic", "cultural representative", "cultural exclusive"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# === Load Wikidata Cache ===
CACHE_PATH = "wikidata_cache_ultra.pkl"
wikidata_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        wikidata_cache = pickle.load(f)
else:
    print("⚠️ Warning: wikidata_cache_ultra.pkl not found. Proceeding without extra metadata.")

# === Load Wikipedia Summary Cache ===
SUMMARY_CACHE_PATH = "wiki_summary_cache.pkl"
summary_cache = {}
if os.path.exists(SUMMARY_CACHE_PATH):
    with open(SUMMARY_CACHE_PATH, "rb") as f:
        summary_cache = pickle.load(f)

# === Helper Functions ===
def get_wikipedia_summary_from_wikidata(wikidata_url, lang="en"):
    if wikidata_url in summary_cache:
        return summary_cache[wikidata_url]
    try:
        wikidata_id = wikidata_url.strip().split("/")[-1]
        response = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json")
        entity = response.json()["entities"][wikidata_id]
        sitelinks = entity.get("sitelinks", {})
        if f"{lang}wiki" in sitelinks:
            title = sitelinks[f"{lang}wiki"]["title"]
            summary_response = requests.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}")
            if summary_response.status_code == 200:
                summary = summary_response.json().get("extract", "")
                summary_cache[wikidata_url] = summary
                return summary
    except Exception as e:
        print(f"Failed to retrieve summary for {wikidata_url}: {e}")
    summary_cache[wikidata_url] = ""
    return ""

def preprocess_dataset(dataset):
    dataset = dataset.filter(
        lambda x: (
            x["description"] is not None and
            x["name"] is not None and
            x["label"] in LABELS
        )
    )

    def enrich(x):
        summary = get_wikipedia_summary_from_wikidata(x["item"])
        info = wikidata_cache.get(x["item"], {})

        metadata_features = (
            f"Attachment: {info.get('attachment', 0)}. "
            f"Spread: {info.get('spread', 0)}. "
            f"Specificity: {info.get('specificity', 0)}. "
            f"Languages: {info.get('n_languages', 0)}. "
            f"Instances: {info.get('n_instanceof', 0)}. "
            f"Subclasses: {info.get('n_subclassof', 0)}. "
            f"DescribedBy: {info.get('n_describedby', 0)}. "
        )

        text = (
            f"{metadata_features} "
            f"Category: {x['category']}. "
            f"Type: {x['type']}. "
            f"Subcategory: {x.get('subcategory', '')}. "
            f"Name: {x['name']}. "
            f"Description: {x['description']}. "
        )
        if summary:
            text += f"Wikipedia Summary: {summary}"

        return {
            "text": text,
            "label": label2id[x["label"]],
        }

    return dataset.map(enrich)

# === Model Setup ===
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

# === Load Dataset ===
raw_dataset = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset")
train_dataset = preprocess_dataset(raw_dataset["train"])
dev_dataset = preprocess_dataset(raw_dataset["validation"])

# Save Wikipedia summary cache
with open(SUMMARY_CACHE_PATH, "wb") as f:
    pickle.dump(summary_cache, f)

# === Class Weights ===
label_counts = Counter(train_dataset["label"])
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / label_counts[i] for i in range(len(LABELS))],
    dtype=torch.float
)

# === Tokenization ===
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Trainer Setup ===
accuracy_metric = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="lm_based/model",
    num_train_epochs=15,                  # ← Increase epochs (was 10)
    per_device_train_batch_size=64,        # ← Reduce batch size a bit (was 128) for better generalization
    per_device_eval_batch_size=64,         # ← Same for eval
    learning_rate=1e-5,                    # ← Lower LR (was 2e-5) for more careful fine-tuning
    warmup_ratio=0.1,                      # ← Slightly higher warmup
    weight_decay=0.01,                     # ← Lower weight decay (was 0.1) to regularize less aggressively
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="lm_based/logs",
    logging_strategy="epoch",
    fp16=True,
    seed=42,
    report_to="none",
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,         # ← Accumulate grads because of smaller batch
    label_smoothing_factor=0.05,           # ← Reduce smoothing (was 0.1) for sharper probability estimates
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === Train and Evaluate ===
trainer.train()
trainer.save_model("lm_based/model")

metrics = trainer.evaluate()
print("Final dev set metrics:", metrics)

# === Plot Learning Curves ===
log_history = trainer.state.log_history
train_loss, train_steps = [], []
eval_loss, eval_accuracy, eval_f1, eval_steps = [], [], [], []

for entry in log_history:
    step = entry.get("step", None)
    if "loss" in entry:
        train_loss.append(entry["loss"])
        train_steps.append(step)
    if "eval_loss" in entry:
        eval_loss.append(entry["eval_loss"])
        eval_accuracy.append(entry.get("eval_accuracy", None))
        eval_f1.append(entry.get("eval_f1", None))
        eval_steps.append(step)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
if train_loss:
    ax[0].plot(train_steps, train_loss, label="Train Loss", marker="o")
if eval_loss:
    ax[0].plot(eval_steps, eval_loss, label="Eval Loss", marker="x")
ax[0].set_title("Training vs Evaluation Loss")
ax[0].set_xlabel("Step")
ax[0].set_ylabel("Loss")
ax[0].legend()

if eval_accuracy:
    ax[1].plot(eval_steps, eval_accuracy, label="Eval Accuracy", marker="s")
if eval_f1:
    ax[1].plot(eval_steps, eval_f1, label="Eval F1", marker="^")
ax[1].set_title("Evaluation Accuracy and F1")
ax[1].set_xlabel("Step")
ax[1].set_ylabel("Score")
ax[1].set_ylim(0, 1)
ax[1].legend()
plt.tight_layout()
plt.savefig("lm_based/training_curves.png")
plt.close()

# === Confusion Matrix ===
predictions = trainer.predict(dev_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

cm = confusion_matrix(labels, preds, labels=[0, 1, 2], normalize="true")
df_cm = pd.DataFrame(cm, index=LABELS, columns=LABELS)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Normalized Confusion Matrix on Dev Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("lm_based/confusion_matrix.png")

# Save classification report
report = classification_report(labels, preds, target_names=LABELS, zero_division=0)
print(report)
with open("lm_based/classification_report.json", "w") as f:
    json.dump(classification_report(labels, preds, target_names=LABELS, output_dict=True), f, indent=2)

print("\n✅ Finished LM training!")
