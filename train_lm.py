# Imports
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding,
    Trainer, set_seed
)
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.nn import CrossEntropyLoss
from evaluate import load as load_metric

# Config
class CFG:
    model_name = "roberta-base"  # Upgraded model
    output_dir = "lm_based/model"
    logging_dir = "lm_based/logs"
    seed = 42
    labels = ["cultural agnostic", "cultural representative", "cultural exclusive"]
    batch_size = 32  # Slightly smaller batch size for stability
    num_epochs = 5
    learning_rate = 3e-5

set_seed(CFG.seed)
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
label2id = {label: i for i, label in enumerate(CFG.labels)}
id2label = {i: label for label, i in label2id.items()}

# Load caches
wikidata_cache = pickle.load(open("wikidata_cache_ultra.pkl", "rb")) if os.path.exists("wikidata_cache_ultra.pkl") else {}
summary_cache = pickle.load(open("wiki_summary_cache.pkl", "rb")) if os.path.exists("wiki_summary_cache.pkl") else {}

# Preprocessing helpers
def build_text(x):
    summary = summary_cache.get(x["item"], "")
    meta = wikidata_cache.get(x["item"], {})
    fields = [
        f"[ATTACHMENT] {meta.get('attachment', 0)}",
        f"[SPREAD] {meta.get('spread', 0)}",
        f"[SPECIFICITY] {meta.get('specificity', 0)}",
        f"[LANGUAGES] {meta.get('n_languages', 0)}",
        f"[INSTANCEOF] {meta.get('n_instanceof', 0)}",
        f"[SUBCLASSOF] {meta.get('n_subclassof', 0)}",
        f"[DESCRIBEDBY] {meta.get('n_describedby', 0)}",
        f"[CATEGORY] {x['category']}",
        f"[TYPE] {x['type']}",
        f"[SUBCATEGORY] {x.get('subcategory', '')}",
        f"[NAME] {x['name']}",
        f"[DESC] {x['description']}",
        f"[WIKI] {summary}"
    ]
    return " | ".join(fields)

def enrich(x):
    return {"text": build_text(x), "label": label2id.get(x["label"], -1)}

def preprocess_dataset(dataset):
    dataset = dataset.filter(lambda x: x.get("description") and x.get("name") and x.get("label") in CFG.labels)
    return dataset.map(enrich, num_proc=4)

# Load and process dataset
raw = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset")
train_dataset = preprocess_dataset(raw["train"])
dev_dataset = preprocess_dataset(raw["validation"])
pickle.dump(summary_cache, open("wiki_summary_cache.pkl", "wb"))

# Tokenization
def tokenize(x):
    return tokenizer(x["text"], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Compute class weights
labels_array = np.array(train_dataset["label"])
present = np.unique(labels_array)
weights_present = compute_class_weight("balanced", classes=present, y=labels_array)
full_weights = np.zeros(len(label2id), dtype=np.float32)
for label, weight in zip(present, weights_present):
    full_weights[label] = weight
class_weights = torch.tensor(full_weights, dtype=torch.float)

# Model and trainer
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.model_name, num_labels=len(CFG.labels), id2label=id2label, label2id=label2id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy_metric = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
    p_r_f = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p_r_f[0], "recall": p_r_f[1], "f1": p_r_f[2]}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = CrossEntropyLoss(weight=class_weights.to(model.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir=CFG.output_dir,
    num_train_epochs=CFG.num_epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    learning_rate=CFG.learning_rate,
    logging_dir=CFG.logging_dir,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(CFG.output_dir)
tokenizer.save_pretrained(CFG.output_dir)
metrics = trainer.evaluate()
print("Final dev set metrics:", metrics)

# Confusion matrix and report
predictions = trainer.predict(dev_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
cm = confusion_matrix(labels, preds, labels=list(label2id.values()), normalize="true")
df_cm = pd.DataFrame(cm, index=CFG.labels, columns=CFG.labels)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Pred")
plt.tight_layout()
plt.savefig("lm_based/confusion_matrix.png")

report = classification_report(labels, preds, target_names=CFG.labels, zero_division=0)
print(report)
with open("lm_based/classification_report.json", "w") as f:
    json.dump(classification_report(labels, preds, target_names=CFG.labels, output_dict=True), f, indent=2)

print("\nDone training.")
