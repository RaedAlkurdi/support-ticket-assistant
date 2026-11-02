# Data preparation

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset

Dataset revision: `430d1a89bd93bd1fa23c16f29dd53e73f0087443`

Supported intents: cancel_order, change_order, change_shipping_address,
track_order, contact_human_agent, get_refund

Split seed: 42

| Split | Rows |
| --- | ---: |
| Train | 4,171 |
| Calibration | 596 |
| Validation | 596 |
| Test | 596 |

Grouping rule: I lowercase each instruction, remove whitespace at the
beginning and end, and replace runs of whitespace with one space.
Identical normalized instructions are kept in the same split.

Data quality checks: There are 0 missing or blank instructions among
the selected rows. There are 1,033 repeated normalized instructions
(counting occurrences beyond the first, not distinct duplicate groups).

Split checks: All rows received a fold. Every split contains all six intents.
No normalized instruction group appears in multiple splits.

Limit: This grouping catches identical normalized messages, but may
miss paraphrases or messages generated from the same template.