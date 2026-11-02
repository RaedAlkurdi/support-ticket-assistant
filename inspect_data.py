from collections import Counter
import re
import numpy as np
from datasets import load_dataset, disable_progress_bars
from sklearn.model_selection import StratifiedGroupKFold

disable_progress_bars()

dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
revision = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
print("Dataset revision:", revision)

dataset = load_dataset(dataset_name, revision=revision)
data = dataset["train"]

intents = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "track_order",
    "contact_human_agent",
    "get_refund",
]

selected_data = data.filter(lambda row: row["intent"] in intents)

missing = sum(
    not isinstance(text, str) or not text.strip()
    for text in selected_data["instruction"]
)
print("Missing or blank instructions:", missing)

# Stop before calling .strip() on a missing value.
if missing:
    raise ValueError("Some selected instructions are missing or blank")

group_ids = []
for row in selected_data:
    group_id = re.sub(r"\s+", " ", row["instruction"].strip().lower())
    group_ids.append(group_id)

print("Repeated normalized instructions:", len(group_ids) - len(set(group_ids)))

splitter = StratifiedGroupKFold(
    n_splits=10, shuffle=True, random_state=42
)
X = np.arange(len(selected_data))
y = selected_data["intent"]

fold_ids = np.full(len(X), -1)

for fold_number, (train_indices, heldout_indices) in enumerate(
    splitter.split(X, y, group_ids)
):
    fold_ids[heldout_indices] = fold_number

fold_to_split = {
    0: "train", 1: "train", 2: "train", 3: "train",
    4: "train", 5: "train", 6: "train",
    7: "calibration", 8: "validation", 9: "test",
}

split_assignments = [fold_to_split[int(fold)] for fold in fold_ids]

assigned_data = selected_data.add_column("row_id", X)
assigned_data = assigned_data.add_column("group_id", group_ids)
assigned_data = assigned_data.add_column("fold_id", fold_ids)
assigned_data = assigned_data.add_column("split", split_assignments)

assigned_data.to_csv("split_assignments.csv", index=False)
assigned_data.select_columns(["row_id", "fold_id", "split"]).to_csv(
    "split_ids.csv", index=False
)

group_to_split = {}
for group_id, split in zip(
    assigned_data["group_id"], assigned_data["split"]
):
    if group_id in group_to_split and group_to_split[group_id] != split:
        print("Group appears in multiple splits:", group_id)

    group_to_split[group_id] = split

for split_name in ["train", "calibration", "validation", "test"]:
    split_data = assigned_data.filter(
        lambda row: row["split"] == split_name
    )
    print(f"\n{split_name}: {len(split_data)} rows")
    print(Counter(split_data["intent"]))

print("Rows without a fold:", np.sum(fold_ids == -1))