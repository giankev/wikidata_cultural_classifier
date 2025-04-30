# === Imports ===
import os
import json
import pickle
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from tqdm import tqdm

from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# === Constants ===
LABELS = ["cultural agnostic", "cultural representative", "cultural exclusive"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}
CACHE_PATH = "wikidata_cache_ultra.pkl"
np.random.seed(42)

# === Load Cache ===
wikidata_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        wikidata_cache = pickle.load(f)

# === Functions ===
def fetch_info(url):
    if url in wikidata_cache:
        return (url, wikidata_cache[url])
    info = {}
    try:
        wikidata_id = url.strip().split("/")[-1]
        response = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json")
        entity = response.json()["entities"][wikidata_id]

        claims = entity.get("claims", {})
        sitelinks = entity.get("sitelinks", {})

        info["attachment"] = sum(1 for p in ["P495", "P2596", "P17"] if p in claims)
        info["spread"] = len([k for k in sitelinks if k.endswith("wiki") and k != "commonswiki"])
        info["specificity"] = len({
            claim["mainsnak"]["datavalue"]["value"]["id"]
            for field in ["P495", "P17"]
            if field in claims
            for claim in claims[field]
            if "datavalue" in claim["mainsnak"]
        })
        info["n_languages"] = len({k.split("wiki")[0] for k in sitelinks if k.endswith("wiki") and k != "commonswiki"})
        info["n_instanceof"] = len(claims.get("P31", []))
        info["n_subclassof"] = len(claims.get("P279", []))
        info["n_describedby"] = len(claims.get("P1343", []))

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        info = {k: 0 for k in ["attachment", "spread", "specificity", "n_languages", "n_instanceof", "n_subclassof", "n_describedby"]}

    return (url, info)

def preload_infos(datasets, num_workers=None):
    urls = set()
    for dataset in datasets:
        urls.update(example["item"] for example in dataset)

    print(f"Preloading {len(urls)} unique Wikidata items with multiprocessing...")
    urls_to_fetch = [u for u in urls if u not in wikidata_cache]

    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 4)

    with multiprocessing.Pool(num_workers) as pool:
        for url, info in tqdm(pool.imap_unordered(fetch_info, urls_to_fetch), total=len(urls_to_fetch)):
            wikidata_cache[url] = info

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(wikidata_cache, f)

    print(f"\u2705 Preloading completed and saved to {CACHE_PATH}")

def preprocess_dataset(dataset):
    dataset = dataset.filter(lambda x: x["description"] is not None and x["name"] is not None and x["label"] in LABELS)

    features = []
    labels = []

    for x in dataset:
        info = wikidata_cache[x["item"]]
        features.append([
            info["attachment"],
            info["spread"],
            info["specificity"],
            info["n_languages"],
            info["n_instanceof"],
            info["n_subclassof"],
            info["n_describedby"],
        ])
        labels.append(label2id[x["label"]])

    return np.array(features), np.array(labels)

def evaluate(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = np.mean(preds == labels)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === Main Process ===
raw_dataset = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset")
preload_infos([raw_dataset["train"], raw_dataset["validation"]])

X_train, y_train = preprocess_dataset(raw_dataset["train"])
X_dev, y_dev = preprocess_dataset(raw_dataset["validation"])

# Grid Search XGBoost
print("Starting training and grid search...")
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [5, 6],
    'learning_rate': [0.03, 0.05],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(objective="multi:softprob", num_class=3, use_label_encoder=False, verbosity=0)
clf = GridSearchCV(xgb, param_grid, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)

print(f"Best parameters: {clf.best_params_}")

# Retrain best model
best_xgb = XGBClassifier(**clf.best_params_, objective="multi:softprob", num_class=3, use_label_encoder=False, verbosity=0)
best_xgb.fit(X_train, y_train)

# Stacking: XGB output -> Logistic Regression
train_proba = best_xgb.predict_proba(X_train)
dev_proba = best_xgb.predict_proba(X_dev)

stacker = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
stacker.fit(train_proba, y_train)

dev_preds = stacker.predict(dev_proba)
metrics = evaluate(dev_preds, y_dev)
print("Final dev set metrics:", metrics)

# === Plotting ===
os.makedirs("nonlm_based", exist_ok=True)
cm = confusion_matrix(y_dev, dev_preds, labels=[0, 1, 2], normalize="true")
df_cm = pd.DataFrame(cm, index=LABELS, columns=LABELS)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Confusion Matrix (Ultra Non-LM Based + Stacking)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("nonlm_based/confusion_matrix_ultra_stacking.png")

# Classification Report
report = classification_report(y_dev, dev_preds, target_names=LABELS, zero_division=0)
print(report)
with open("nonlm_based/classification_report_ultra_stacking.json", "w") as f:
    json.dump(classification_report(y_dev, dev_preds, target_names=LABELS, output_dict=True), f, indent=2)

print("\n\u2705 Ultra Stacking Mode Completed!")
