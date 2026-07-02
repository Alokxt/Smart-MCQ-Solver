import os 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import string 
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
sem = SentenceTransformer("all-MiniLM-L6-v2")
def encode_question_and_options(df, batch_size=64):
    q_texts = [str(row["prompt"]) for _, row in df.iterrows()]
    opt_texts = []
    for _, row in df.iterrows():
        for c in CHOICES:
            opt_texts.append(str(row[c]))

    q_embs = sem.encode(q_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    opt_embs = sem.encode(opt_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)

    opt_embs = opt_embs.reshape(len(df), 5, EMB_DIM)
    return q_embs, opt_embs

df = pd.read_csv("path")
all_q_embs, all_opt_embs = encode_question_and_options(df)

class MCQDataset(Dataset):
    def __init__(self, q_embs, opt_embs, labels):
        self.q_embs   = torch.tensor(q_embs, dtype=torch.float32)
        self.opt_embs = torch.tensor(opt_embs, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, i):
        return self.q_embs[i], self.opt_embs[i], self.labels[i]

def get_train_val_Loader(train_idx,val_idx):
    train_ds = MCQDataset(all_q_embs[train_idx], all_opt_embs[train_idx], df["label"].values[train_idx])
    val_ds   = MCQDataset(all_q_embs[val_idx],   all_opt_embs[val_idx],   df["label"].values[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
