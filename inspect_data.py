from datasets import load_dataset, disable_progress_bars
from collections import Counter
import re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
disable_progress_bars()

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

data = dataset["train"]
# cancel_order, change_order, change_shipping_address, track_order, contact_human_agent, get_refund

intents = ["cancel_order", "change_order", "change_shipping_address", "track_order", "contact_human_agent", "get_refund"]

examples_by_intent = {intent: [] for intent in intents}
selected_data = data.filter(lambda row: row["intent"] in intents)


group_ids = []
for row_id, row in enumerate(selected_data):
    group_id = re.sub(r"\s+", " ", row["instruction"].strip().lower())    
    group_ids.append(group_id)


splitter = StratifiedGroupKFold(n_splits= 10, shuffle=True, random_state=42)
X = np.arange(len(selected_data))
y = selected_data["intent"]
groups = group_ids

fold_ids = np.full(len(X), -1)

for fold_number, (train_indices, heldout_indices) in enumerate(
    splitter.split(X, y, groups)
):
    fold_ids[heldout_indices] = fold_number

fold_to_split = {
    0: "train", 1: "train", 2: "train", 3: "train",
    4: "train", 5: "train", 6: "train",
    7: "calibration", 8: "validation", 9: "test"
}

split_assignments = [fold_to_split[int(fold)] for fold in fold_ids]

assigned_data = selected_data.add_column("row_id", X)
assigned_data = assigned_data.add_column("group_id", group_ids)
assigned_data = assigned_data.add_column("fold_id", fold_ids)
assigned_data = assigned_data.add_column("split", split_assignments)

assigned_data.to_csv(
    r"C:\Users\alkur\Desktop\Complex Adaptive Systems\Own Projects\support-ticket-assistant\split_assignments.csv"
)

group_to_split = {}
for group_id, split in zip(
    assigned_data["group_id"], assigned_data["split"]
):
    if group_id in group_to_split and group_to_split[group_id] != split:
        print("Group appears in multiple splits:", group_id)
    
    group_to_split[group_id] = split

for split_name in ["train", "calibration", "validation", "test"]:
    split_data = assigned_data.filter(lambda row: row["split"] == split_name)

    print(f"\n{split_name}: {len(split_data)} rows")
    print(Counter(split_data["intent"]))

print("Rows without a fold:", np.sum(fold_ids == -1))