from data.dataloaders import get_test_loader
from models.deBerta import get_debertaLora
import torch 
import pandas as pd 
import numpy as np 

WEIGHTS_PATH = ""
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 192
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 15
LR = 3e-5
SAVE_PATH = "deberta_base_lora.pt"

model = get_debertaLora()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
model = model.to(device)
model.eval()
test_loader = get_test_loader()

OPTION_COLS = ["A","B","C","D","E"]
test_df = pd.read_csv("path")


all_probs = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
        all_probs.extend(probs)
        if (i + 1) % 20 == 0:
            print(f"  Processed {(i+1)*BATCH_SIZE}/{len(test_df)} rows")

predictions = []
for probs in all_probs:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_labels = " ".join([OPTION_COLS[i] for i in top3_idx])
    predictions.append(top3_labels)

id_col = test_df["id"]

submission = pd.DataFrame({"ID": id_col, "Prediction": predictions})
submission.to_csv("submission.csv", index=False)

print(f"Total predictions: {len(submission)}")
print(submission.head(10))