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


MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 192
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 15
LR = 3e-5
SAVE_PATH = "deberta_base_lora.pt"

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["query_proj", "key_proj", "value_proj", "out_proj"],
    bias="none"
)

base_model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float32
)


def get_debertaLora():

    base_model.gradient_checkpointing_enable()

    model = get_peft_model(base_model, lora_config)

    model.load_state_dict(
        torch.load(wt_path),
        strict=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model 

    
            
