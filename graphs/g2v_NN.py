# Train a neural network using graphs features, embedding and
# the two features 'number_sitelinks', 'sitelinks_translation_entropy'
# Some validation results and tests saved in train_results.txt

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------- Feature Toggle --------
USE_GRAPH    = True
USE_EMBED    = True
USE_SITELINK = True

# -------- Configuration --------
TRAIN_CSV = os.path.expanduser('~/content/train_g2v.csv')
VALID_CSV = os.path.expanduser('~/content/valid_g2v.csv')
EMBED_DIR = os.path.expanduser('~/content/graph2vec_out/')
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_DIM = 128
LABEL2IDX = {
    'cultural agnostic': 0,
    'cultural exclusive': 1,
    'cultural representative': 2
}
NUM_CLASSES = len(LABEL2IDX)

# -------- Data Loading & Preprocessing --------
def load_df(path):
    df = pd.read_csv(path)
    df = df[df['label'].isin(LABEL2IDX)]
    df['target'] = df['label'].map(LABEL2IDX)
    return df

df_train = load_df(TRAIN_CSV)
unique, counts = np.unique(df_train['target'], return_counts=True)
class_counts = dict(zip(unique, counts))
class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
weights = [class_weights[i] for i in range(NUM_CLASSES)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

df_val = load_df(VALID_CSV)

data_cols = set(df_train.columns) - {'item','qid','language','name','description','type','category','subcategory','label','target','g2v_valid'}

graph_feats = [col for col in data_cols if 'degree' in col or 'clustering' in col or 'pagerank' in col or 'centrality' in col]
sitelink_feats = ['number_sitelinks', 'sitelinks_translation_entropy']
other_feats = list(data_cols - set(graph_feats) - set(sitelink_feats))

feature_cols = []
if USE_GRAPH:
    feature_cols += graph_feats
if USE_SITELINK:
    feature_cols += sitelink_feats
feature_cols += other_feats

def clean_df(df):
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

clean_df(df_train)
clean_df(df_val)

scaler = StandardScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_val[feature_cols] = scaler.transform(df_val[feature_cols])

# -------- Dataset & DataLoader --------
class CulturalDataset(Dataset):
    def __init__(self, df, embed_dir, feature_cols, use_embed=True):
        self.features = df[feature_cols].values.astype(np.float32)
        self.qids = df['qid'].tolist()
        self.valid = df['g2v_valid'].tolist()
        self.targets = df['target'].values.astype(np.int64)
        self.embed_dir = embed_dir
        self.use_embed = use_embed

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.use_embed:
            qid = self.qids[idx]
            emb_path = os.path.join(self.embed_dir, f"{qid}.npy")
            if self.valid[idx] and os.path.exists(emb_path):
                emb = np.load(emb_path).astype(np.float32)
            else:
                emb = np.zeros(EMBED_DIM, dtype=np.float32)
            return torch.from_numpy(feat), torch.from_numpy(emb), torch.tensor(self.targets[idx])
        else:
            return torch.from_numpy(feat), torch.tensor(self.targets[idx])

train_ds = CulturalDataset(df_train, EMBED_DIR, feature_cols, use_embed=USE_EMBED)
val_ds = CulturalDataset(df_val, EMBED_DIR, feature_cols, use_embed=USE_EMBED)

def collate_fn(batch):
    if USE_EMBED:
        feats, embs, labels = zip(*batch)
        return (torch.stack(feats).to(DEVICE), torch.stack(embs).to(DEVICE)), torch.tensor(labels, dtype=torch.long, device=DEVICE)
    else:
        feats, labels = zip(*batch)
        return torch.stack(feats).to(DEVICE), torch.tensor(labels, dtype=torch.long, device=DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------- Model Definition --------
class TwoBranchNN(nn.Module):
    def __init__(self, feat_dim, emb_dim, hidden_dim=128, num_classes=NUM_CLASSES, drop=0.3, use_embed=True):
        super().__init__()
        self.use_embed = use_embed
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(drop)
        )
        if self.use_embed:
            self.emb_net = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(drop)
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x):
        if self.use_embed:
            feat, emb = x
            f = self.feat_net(feat)
            e = self.emb_net(emb)
            h = torch.cat([f, e], dim=1)
        else:
            feat = x
            h = self.feat_net(feat)
        return self.classifier(h)

feat_dim = len(feature_cols)
model = TwoBranchNN(feat_dim, EMBED_DIM, use_embed=USE_EMBED).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# -------- Training Loop --------
best_loss = float('inf')
patience_ctr = 0

print(f"Training on {DEVICE}")
start = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        if USE_EMBED:
            (feats, embs), labels = batch
            inputs = (feats, embs)
        else:
            feats, labels = batch
            inputs = feats

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            if USE_EMBED:
                (feats, embs), labels = batch
                inputs = (feats, embs)
            else:
                feats, labels = batch
                inputs = feats

            logits = model(inputs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    val_loss /= len(val_ds)
    scheduler.step(val_loss)
    val_acc = accuracy_score(trues, preds)
    val_f1 = f1_score(trues, preds, average='macro', zero_division=0)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
end = time.time()
print(f"Training done in {(end-start):.1f}s")

# load best
model.load_state_dict(torch.load('best_model.pth'))

# -------- Final Evaluation --------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        if USE_EMBED:
            (feats, embs), labels = batch
            inputs = (feats, embs)
        else:
            feats, labels = batch
            inputs = feats

        logits = model(inputs)
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

acc = accuracy_score(trues, preds)
precision = precision_score(trues, preds, average='macro', zero_division=0)
recall = recall_score(trues, preds, average='macro', zero_division=0)
f1 = f1_score(trues, preds, average='macro', zero_division=0)
cm = confusion_matrix(trues, preds)

print("\nFinal Evaluation on validation set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL2IDX.keys(), yticklabels=LABEL2IDX.keys())
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Pred')
plt.tight_layout()
plt.show()
