
model = get_peft_model(base_model, lora_config)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
model = model.to(device)
model.eval()
print("Model loaded successfully.")



test_ds = MCQTestDataset(test_df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

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