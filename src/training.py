import wandb 
from models.deBerta import get_debertaLora 
from data.dataloaders import get_train_val_loader
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
import numpy as np 

import torch 
EPOCHS =15 

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 192
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 15
LR = 3e-5
SAVE_PATH = "deberta_base_lora.pt"

model = get_debertaLora()
for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float
train_loader, val_loader  = get_train_val_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01
    )
steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, total_steps // 10)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
def map3_loss(logits, labels):
    ce = F.cross_entropy(logits.float(), labels)
    top3 = torch.topk(logits, k=3, dim=-1).indices
    in_top3 = (top3 == labels.unsqueeze(1)).any(dim=1).float()
    recall_penalty = (1 - in_top3).mean()
    return ce + 0.5 * recall_penalty

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

run = wandb.init(
    project="23f2001025-t22026",       
    entity="23f2001025-indian-institue-of-technology-madras",  
    name="deberta-v3-base",       
    config={
        "model": "deberta-v3-base",
        "Optim":"AdamW",
        "epochs": 17,
        "batch_size": 2,
        "lr": 3e-6
    }
)
best_map3 = 0.0
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        loss = map3_loss(logits, labels) / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    if len(train_loader) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.extend(outputs.logits.float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_map3 = compute_map3(all_logits, all_labels)
    avg_loss = total_loss / max(1, len(train_loader))
    wandb.log({
       "avg_loss":avg_loss,
       "val_map3":val_map3,
       
    })
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val MAP@3: {val_map3:.4f}")
    if val_map3 > best_map3:
        best_map3 = val_map3
        torch.save(model.state_dict(), "deberta_base_lora7.pt")
        print(f"  Saved best model with MAP@3={best_map3:.4f}")
    

