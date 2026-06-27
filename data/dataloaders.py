import os 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import string 


TRAIN_PATH = ""
TEST_PATH = ""

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
QUESTION_COL = "prompt2"
ANSWER_COL = "answer"
MODEL_NAME = "microsoft/deberta-v3-base"

df = pd.read_csv(TRAIN_PATH)
df.columns = [c.strip() for c in df.columns]
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL_MAP = {v: i for i, v in enumerate(OPTION_COLS)}

df["label_idx"] = df[ANSWER_COL].str.strip().map(LABEL_MAP)
df = df.dropna(subset=["label_idx"])
df["label_idx"] = df["label_idx"].astype(int)
df["prompt2"] = df["prompt"].apply(lambda x:"".join(ch for ch in x if ch not in string.punctuation))

train_df, val_df = train_test_split(df, test_size=0.1, random_state=7, stratify=df["label_idx"])


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MCQDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = str(row[QUESTION_COL])
        input_ids_list, attn_mask_list = [], []
        for col in OPTION_COLS:
            enc = self.tokenizer(
                question,
                str(row[col]),
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))
        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels": label
        }
    
test_df = pd.read_csv(TEST_PATH)
test_df.columns = [c.strip() for c in test_df.columns]
test_df["prompt2"] = test_df["prompt"].apply(lambda x:"".join(ch for ch in x if ch not in string.punctuation))
print("Test columns:", test_df.columns.tolist())
print("Test shape:", test_df.shape)
class MCQTestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = str(row[QUESTION_COL])
        input_ids_list, attn_mask_list = [], []
        for col in OPTION_COLS:
            enc = self.tokenizer(
                question,
                str(row[col]),
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
        }

    
def get_train_val_loader():
    train_ds = MCQDataset(train_df, tokenizer, MAX_LEN)
    val_ds = MCQDataset(val_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader , val_loader

def get_test_loader():
    