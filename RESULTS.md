# Results

## Day 2: TF-IDF intent classification baseline

### Setup

- Dataset: Bitext customer support dataset, revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`
- Supported intents: `cancel_order`, `change_order`, `change_shipping_address`, `track_order`, `contact_human_agent`, `get_refund`
- Training rows: 4,171
- Validation rows: 596
- Model: `TfidfVectorizer` followed by `LogisticRegression`
- The pipeline was fitted on training rows and evaluated on validation rows.

### Majority-class reference

The most frequent training intent was `contact_human_agent` (999 examples). Predicting it for every validation message gives 100 correct out of 596, or **16.78% accuracy**.

### Validation results

- Accuracy: **99.33%** (592/596)
- Macro precision: 0.99
- Macro recall: 0.99
- Macro F1: 0.99
- Weighted precision: 0.99
- Weighted recall: 0.99
- Weighted F1: 0.99

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cancel_order | 0.99 | 1.00 | 0.99 | 99 |
| change_order | 0.97 | 1.00 | 0.99 | 100 |
| change_shipping_address | 1.00 | 1.00 | 1.00 | 98 |
| contact_human_agent | 1.00 | 1.00 | 1.00 | 100 |
| get_refund | 1.00 | 1.00 | 1.00 | 100 |
| track_order | 1.00 | 0.96 | 0.98 | 99 |

### Misclassified validation examples

All four errors had the true intent `track_order`:

- Three messages using the short template `trackorder {{Order Number}}` were predicted as `change_order`.
- One message containing the typo `trackng purchase` was predicted as `cancel_order`.

Copy the full message text for each row from the `mistakes` table in the notebook if you want the exact examples here.

### Notes

The score is very high on this validation split. The data is synthetic, and grouping only identical normalized instructions may leave similar templates across splits. This result measures performance on this dataset split and does not establish performance on real customer messages.

The fitted pipeline was saved to `artifacts/tfidf_baseline.joblib`. Predictions from the loaded pipeline matched the original model's predictions on the validation messages.