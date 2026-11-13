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

## Day 3: DistilBERT fine-tuning and comparison

### Setup
- Model: `distilbert/distilbert-base-uncased`, sequence classification head with six labels
- Training rows: 4,171
- Validation rows: 596
- Epochs: 2
- Learning rate: 2e-5
- Batch size: 16
- Weight decay: 0.01
- Training steps: 522
- Training runtime: 410.83 seconds
- Device: [CPU or GPU and model]
- Seed: [record the seed used]

### Validation results
- Accuracy: **99.83%** (595/596)
- Evaluation loss: **0.0143**
- TF-IDF baseline accuracy: **99.33%** (592/596)
- DistilBERT made one validation error; TF-IDF made four. The models’ errors were different.

### Error review
DistilBERT misclassified one `change_order` message (“can I edi order {{Order Number}}?”) as `track_order`. TF-IDF classified that message correctly.

TF-IDF mislabeled four `track_order` messages as `change_order` or `cancel_order`. DistilBERT classified all four correctly. Three of those examples use the same instruction, `trackorder {{Order Number}}`; the fourth contains the typo `trackng purchase`.

### Approximate inference timing
One timing run over 596 messages:
- DistilBERT `Trainer.predict` on already-tokenized validation data: **8.62 seconds** (~14.5 ms/message)
- TF-IDF pipeline `predict` on raw validation text: **0.021 seconds** (~0.035 ms/message)

These timings are approximate and not directly comparable: DistilBERT tokenization was outside the timed section, while TF-IDF vectorization was included. `Trainer.predict` also includes batching and Trainer overhead. This was a single run, without repeated measurements.

### Saved model check
Saved the fine-tuned model and tokenizer to `artifacts/distilbert_intents`. Reloaded them and confirmed the model returned six intent scores for a new message.

### Limitations
Results are from the synthetic Bitext dataset and one validation split. Identical-message grouping does not catch all paraphrases or shared templates. The test split remains unused.

## Day 4: Calibration and review routing

### Temperature scaling

I fitted a temperature-scaling parameter on the calibration split to adjust
DistilBERT's predicted probabilities. The fitted temperature was **0.67236**.

| Dataset | Uncalibrated log loss | Calibrated log loss |
|---|---:|---:|
| Calibration | 0.05729 | 0.03373 |
| Validation | 0.06035 | 0.03264 |

The calibrated probabilities had lower log loss on both splits. I also generated
a reliability plot comparing calibrated and uncalibrated confidence on the
validation split.

### Confidence thresholds

I examined thresholds of 0.70, 0.80, 0.90, and 0.95. The saved notebook
results used confidence and correctness values from the **calibration split**,
not the validation split.

| Threshold | Accepted | Sent to review | Incorrectly accepted | Coverage | Accepted-error rate |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 592 | 4 | 2 | 99.33% | 0.34% |
| 0.80 | 591 | 5 | 2 | 99.16% | 0.34% |
| 0.90 | 586 | 10 | 2 | 98.32% | 0.34% |
| 0.95 | 571 | 25 | 2 | 95.81% | 0.35% |

These threshold results used the uncalibrated confidence scores. Validation
threshold metrics for calibrated probabilities still need to be calculated.

### Development challenge check

I applied temperature scaling to the DistilBERT logits using the fitted
temperature. With a threshold of **0.80**, messages with a maximum probability
greater than 0.80 were accepted automatically; the rest were sent for manual
review. The `expected_outcome` column was used only to compare results after
routing.

Of the 10 development challenge messages, 6 were accepted and 4 were sent for
review. Three accepted messages matched the expected outcome. The other three
should have gone to review or received a different intent. This gives an
accepted-error rate of **3/6 (50%)** and coverage of **6/10 (60%)** on this
small challenge set.

| Row | Expected outcome | Routing result |
|---:|---|---|
| 0 | `track_order` | Manual review |
| 1 | `change_order` | Accepted as `change_order` |
| 2 | `change_shipping_address` | Accepted as `change_shipping_address` |
| 3 | `contact_human_agent` | Accepted as `contact_human_agent` |
| 4 | `human_review` | Accepted as `get_refund` |
| 5 | `human_review` | Manual review |
| 6 | `cancel_order` | Accepted as `change_shipping_address` |
| 7 | `track_order` | Manual review |
| 8 | `human_review` | Accepted as `contact_human_agent` |
| 9 | `human_review` | Manual review |

Rows 4, 6, and 8 were incorrect automatic acceptances. Rows 0 and 7 were sent
to review even though the expected outcome named a supported intent. Rows 5
and 9 correctly went to review.

## Day 5: Final evaluation

I evaluated both saved models on 596 held-out test messages.

| Model | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 592/596 | 99.33% | 0.99 |
| DistilBERT | 591/596 | 99.16% | 0.99 |

With temperature scaling and the chosen threshold of 0.90, DistilBERT
accepted 586 test messages and sent 10 to review. One accepted prediction
was wrong, giving 98.32% coverage and a 0.17% accepted-error rate.

The accepted error was a cancellation request predicted as `get_refund`
with 97.88% confidence.

### Final challenge set

Of 10 messages, seven were accepted and three were sent to review. Three
of the seven accepted messages were wrong. Two of the four cases expected
to go to review were accepted automatically.

### Saved inference

Both saved models loaded and predicted in a fresh process. Messages below
or equal to the threshold were appended to a local review queue with their
suggested intent and confidence.

These results come from a synthetic test set and a ten-message challenge
set. They do not establish performance on real customer messages.