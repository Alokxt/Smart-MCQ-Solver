from data.dataloader2 import get_train_val_Loader
from models.Lstm import get_lstm
import torch 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import pandas as pd 
CHOICES     = list("ABCDE")
LABEL_MAP   = {v: i for i, v in enumerate(CHOICES)}
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 5e-4
PATIENCE    = 5
EMB_DIM     = 384
HIDDEN_DIM  = 64
NUM_LAYERS  = 1
N_FOLDS     = 5
WEIGHT_DECAY = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_map3(logits_all, labels_all):
    scores = []
    for logits, label in zip(logits_all, labels_all):
        top3 = np.argsort(logits)[::-1][:3].tolist()
        score = 0.0
        for rank, pred in enumerate(top3):
            if pred == int(label):
                score = 1.0 / (rank + 1)
                break
        scores.append(score)
    return float(np.mean(scores))
def train_one_fold(train_idx, val_idx, fold_num):
    
    train_loader , val_loader = get_train_val_Loader(train_idx,val_idx)
   
 
    model = get_lstm()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
    best_map3 = 0.0
    best_state = None
    patience_counter = 0
 
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for q_emb, opt_embs, labels in train_loader:
            q_emb, opt_embs, labels = q_emb.to(device), opt_embs.to(device), labels.to(device)
            logits = model(q_emb, opt_embs)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.15)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
 
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for q_emb, opt_embs, labels in val_loader:
                logits = model(q_emb.to(device), opt_embs.to(device))
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.numpy())
 
        val_map3 = compute_map3(all_logits, all_labels)
 
        if val_map3 > best_map3:
            best_map3 = val_map3
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
 
    print(f"Fold {fold_num} | Best Val MAP@3: {best_map3:.4f}")
    

    model.load_state_dict(best_state)
   
   
    
    return model, best_map3
 
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_models = []
fold_scores = []

df = pd.read_csv("path")
 
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    model, score = train_one_fold(train_idx, val_idx, fold + 1)
    fold_models.append(model)
    fold_scores.append(score)
    torch.save(model.state_dict(), f"lstm_fold{fold+1}.pt")