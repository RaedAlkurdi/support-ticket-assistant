# Data preparation

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset

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

Checks: All rows received a fold. Every split contains all six intents.
No normalized instruction group appears in multiple splits.

Limit: This grouping catches identical normalized messages, but may
miss paraphrases or messages generated from the same template.