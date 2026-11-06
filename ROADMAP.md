# Support Ticket Assistant: a smaller 10-day roadmap

One project, four working milestones. Updated November 1, 2026.

This replaces the 16-day roadmap with a smaller first version: **10 study days, about 3-4 hours per day**, including breaks, learning, and notes. It is a target of roughly 30-40 hours, not a promise that unfamiliar topics will fit a deadline. Keep the daily limit and add a session if a milestone needs it.

Your environment setup is already reported complete. Reuse that work and your inspection script; there is no need to restart.

## What gets smaller

| Area | Smaller first version |
| --- | --- |
| Classification scope | Six actual intent labels from the dataset, selected once |
| Training data | Initially about 100-200 training messages per intent |
| Models | TF-IDF + logistic regression versus one small pretrained encoder |
| Experiments | One short encoder run; no search across models or hyperparameters |
| Uncertainty | One calibration method and one review threshold |
| Help center | Eight short articles for a fictional shop |
| RAG evaluation | Ten development questions and ten final questions |
| Workflow | A fixed LangGraph flow with local mock tickets |
| Interface | Command line; no web UI, deployment, or real integrations |
| Documentation | One short results document with a section for each milestone |

All four parts stay: model comparison, calibrated uncertainty routing, retrieval with cited drafts, and explicit approval with persistence and retry checks. Larger datasets, more intents, extensive tuning, hybrid retrieval, rerankers, autonomous agents, and interface polish can follow after this version is finished.

Choosing fewer intents makes this a narrower classification task. It does not demonstrate performance on excluded intents or real customer traffic. The application must explain its supported scope.

## Daily rhythm

- Learn: 30 minutes of selected documentation or video.
- Build: 60 minutes.
- Break: 10 minutes.
- Build: 60 minutes.
- Break: 10 minutes.
- Check outputs and record results: 30 minutes.

That is **3 hours of focused work plus 20 minutes of breaks**. Allow up to 40 extra minutes if useful, with a four-hour total ceiling. Split the work across the day if preferred. Training can run unattended; waiting for it does not require extra focus time.

Do not watch entire courses. If a concept needs longer, use a build block for it and carry one task forward. Stop at the daily limit rather than rushing the checks.

## Overview

| Day | Focus | Finish with |
| --- | --- | --- |
| 1 | Inspect and prepare a small dataset | Six intents, fixed splits, a data note |
| 2 | Train the classical baseline | Saved pipeline and validation results |
| 3 | Fine-tune and compare one encoder | Part 1: model comparison and saved inference |
| 4 | Calibrate probabilities and add review routing | One measured routing rule and a review queue |
| 5 | Evaluate and document Parts 1-2 | Part 2: held-out results and working routing |
| 6 | Create articles and compare retrieval | Eight articles and a small retrieval experiment |
| 7 | Add cited reply drafting | Evidence-based drafts or insufficient information |
| 8 | Evaluate and finish RAG | Part 3: results and an inspectable demonstration |
| 9 | Add LangGraph and human approval | A fixed workflow with saved review state |
| 10 | Check retries and package the demo | Part 4: verified mock updates and run instructions |

A milestone counts as finished when its checks pass. Preserve each milestone in the same repository.

---

## Day 1 - Prepare a small dataset

**Goal:** understand the rows and make evaluation repeatable.

**Read:** [Bitext dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), focusing on fields, generation, labels, and terms; [loading a Hugging Face dataset](https://huggingface.co/docs/datasets/loading) as needed.

1. Continue your inspection script. Print columns, row count, example messages, and label counts.
2. Choose six actual intent labels once, before comparing models. Read a few examples per label. Use customer instruction as input and intent as target; exclude response, category, and generation flags.
3. Check missing text and duplicates. Group matching messages after simple case/whitespace/placeholder normalization so groups stay in one split. Inspect a few paraphrases and document that this does not catch every shared template.
4. Save approximately 70% training, 10% calibration, 10% validation, and 10% test assignments with stable IDs and a seed, preserving label coverage where possible.
5. If needed, reduce only the training portion to roughly 100-200 examples per intent. Both models must use the same selected training IDs; keep the held-out assignments fixed.
6. Write ten development challenge messages and ten separate final challenge messages, including ambiguous/unrelated requests. Give clear messages their supported intent and ambiguous/unrelated cases an expected review action. Seal the final set until Day 5.

**Keep:** split IDs, dataset revision, selected labels, grouping limits, and a short data note.

**Done:** you can explain what each split is for, the six supported intents, and why synthetic-data results have limits. If split preparation needs another session, finish it before training.

## Day 2 - Train the baseline

**Goal:** establish a useful model with few moving parts.

**Read:** [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) for TF-IDF; [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for combining preprocessing and classification. Reuse your existing precision/recall knowledge.

1. Record a trivial most-frequent-label reference.
2. Fit one TF-IDF + logistic-regression pipeline on training data only.
3. Evaluate validation accuracy, macro-F1, per-class precision/recall, and a confusion matrix.
4. Read five mistakes, or all if fewer. Explain one likely cause.
5. Save the whole pipeline and label order. Reload in a new process and classify new text.

**Keep:** baseline results, examples, and the saved local artifact.

**Done:** inference works without retraining and no vocabulary fitting used held-out text.

## Day 3 - Fine-tune one encoder and compare

**Goal:** connect your Transformer knowledge to pretrained-model classification.

**Watch/read:** the selected lessons on [tokenizers](https://huggingface.co/learn/llm-course/chapter2/4) and [padding/attention masks](https://huggingface.co/learn/llm-course/chapter2/5), with embedded videos. Use the [text-classification tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification) as your implementation reference. Spread lessons into build time if needed.

1. Inspect token IDs and attention masks for two messages. Create a stable intent-to-ID mapping.
2. Run a tiny training/save/reload smoke test with one small encoder, such as DistilBERT.
3. Complete one short fine-tuning run, initially one epoch on the selected training subset. Record configuration, model revision, seed, hardware, and time. Do not chase a target score.
4. Compare with the baseline on the same validation examples: macro-F1, per-class metrics, and approximate inference latency under the same conditions.
5. Inspect about ten mistakes across the models. Save the encoder and tokenizer; verify fresh-process prediction. Select a routing model using the evidence, even if the baseline wins.

**Hardware limit:** spend at most 30-45 minutes investigating acceleration. Use a compatible cloud runtime if available or carry training forward. Do not turn hardware setup into another project.

**Part 1 complete:** both models have a fair validation comparison, saved inference, and a short failure explanation. Final test evaluation waits until Day 5.

## Day 4 - Calibrate and route to review

**Goal:** measure the probabilities before using them to accept classifications.

**Read:** [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), focusing on reliability plots, multiclass behavior, and one appropriate calibration method.

1. Freeze the selected model. Fit one calibrator only on the calibration split, keeping classifier weights fixed.
2. On validation data, compare uncalibrated/calibrated outputs. Use a small top-label reliability plot: bin highest class scores and compare average score to fraction correct, with bin counts. Do not feed multiclass labels into an ordinary binary calibration example.
3. Try a few review thresholds. Record accepted count, wrong accepted count, coverage, review rate, and accepted-error rate.
4. Use a chosen development target, such as observed accepted-error rate at most 5%, only if meaningful sample counts and useful coverage support it. Report when it fails. No accepted examples means undefined error rate, not success.
5. Add a small local queue with ticket ID, text, suggested intent, score, and review reason. Try only the development challenge messages.

Coverage is the fraction accepted automatically for classification; accepted-error rate is errors among those accepted. Compare before/after calibration at similar coverage. A high class score does not reliably detect unrelated messages.

**Done:** a saved calibrator and threshold produce inspectable accept/review decisions. If this takes longer, use Day 5's time to finish it before final evaluation.

## Day 5 - Finish and evaluate Parts 1-2

**Goal:** measure frozen decisions and make the first release runnable.

**Reference:** [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics). Use today's learning block to resolve gaps rather than add a new topic.

1. Freeze both models, preprocessing, calibrator, routing threshold, and label mapping.
2. Evaluate both classifiers on final test data and evaluate routing for the chosen model. Open the final challenge set and report it separately.
3. Keep ambiguous/unrelated cases out of ordinary single-intent accuracy; report whether they reached review. Document confidently wrong failures.
4. Write concise classification and routing sections in RESULTS.md: metrics, counts, latency conditions, failures, and limitations.
5. Verify a fresh process can predict and queue a message from saved artifacts. Document actual commands.

**Part 2 complete:** model comparison, probability assessment, routing behavior, and held-out results exist. Do not tune against final test failures and still call the result held out.

## Day 6 - Create eight articles and compare retrieval

**Goal:** retrieve relevant policy evidence before generating answers.

**Watch:** introductory indexing and retrieval lessons in [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x), with [official companion notebooks](https://github.com/langchain-ai/rag-from-scratch). **Read:** [Sentence Transformers quickstart](https://sbert.net/docs/quickstart.html) for encoding and similarity. These are selected lessons, not whole-course prerequisites.

1. Write eight short fictional-shop articles with stable IDs, titles, and clear policies. Cover the chosen intents where practical and explicitly list gaps.
2. Create ten development questions and ten separate final questions, including three unanswerable cases in each set. Label relevant article IDs and expected answerability before viewing results. Seal final questions until Day 8.
3. Build a TF-IDF article-similarity baseline, separate from the classification vectorizer.
4. Encode articles with one small sentence-embedding model and compare retrieval on development questions. A matrix of vectors is sufficient; check model length limits and avoid silent truncation.
5. Record hit rate@3 on answerable questions: the fraction with a relevant article among the first three. Inspect failures and choose one retriever.

**Done:** ranked articles can be inspected and the keyword/embedding comparison is recorded. Keep evaluation questions and reference answers outside the indexed corpus. Do not hard-filter articles by the predicted intent.

## Day 7 - Add reply drafting

**Goal:** turn retrieved evidence into a suggested answer.

**Learn:** the introductory generation section of [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch). For a hosted option, consult the [Hugging Face inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference), or use the current documentation for a model/runtime you already have.

1. Choose one existing instruction-following model and verify a small request works. Record name/version, prompt, and cost if applicable. Free hosted access or suitable local hardware is not assumed.
2. Provide the message and retrieved passages, including article IDs. Request a concise draft with citations or an insufficient-information result.
3. Return draft text, status, citation IDs, and supporting passages together.
4. Try development questions, inspecting missing evidence, unsupported answers, and incorrect refusals. A real article ID alone is not proof of support.
5. Keep instructions separate from message/article content. Never let a draft claim a refund or account update has actually happened.

**Done:** a reviewer can inspect draft and evidence together. If model access is unavailable, mark drafting incomplete instead of substituting canned replies and calling it RAG.

## Day 8 - Evaluate and finish Part 3

**Goal:** check evidence and answers on the held-out questions.

**Learn:** review retrieval failures versus generation failures. No new framework today.

1. Freeze the knowledge base, retriever, reply model settings, and prompt.
2. Open the ten final questions. Evaluate both retrieval methods and the selected answering pipeline.
3. For each answer, record policy correctness, factual support, citation accuracy, and appropriate refusal. Report answerable/unanswerable counts separately, with observed latency and cost.
4. Distinguish wrong retrieval from wrong generation despite sufficient evidence. State if you authored and graded the evaluation; these small sets are exploratory.
5. Add a short RAG section to RESULTS.md and demonstrate one supported answer and one insufficient-information case.

**Part 3 complete:** retrieval comparison, actual drafting, citations, refusal behavior, and recorded evaluation. Fix unmet checks before adding the workflow.

## Day 9 - Add LangGraph and approval

**Goal:** coordinate your existing functions with explicit human decisions.

**Watch:** selected Simple Graph, Router, State Schema, and human-feedback lessons in [Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph). **Read:** current [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) and [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) documentation. Favor current APIs over older video syntax.

1. Try a tiny graph with fake outputs, then wrap the existing classifier, retriever, and drafter without rewriting them.
2. Carry ticket ID, message, intent, review status, evidence, draft version, decision, and update status through the graph.
3. Pause uncertain classifications for confirmation/correction before continuing. Route missing evidence to review without an unsupported reply.
4. Pause every draft for approve, edit-and-approve, or reject. Bind approval to the exact displayed or edited version.
5. Add a supported durable local checkpointer, such as a SQLite-backed option. Confirm that a pending ticket survives process exit and resumes using the same identity.

**Done:** the graph reaches the right review states and can persist/resume them. Keep mock reply updates disabled until approval behavior works.

## Day 10 - Verify safe mock updates and finish

**Goal:** complete the smallest reliable human-approved workflow and demonstration.

**Reference:** [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html) if automated checks are new. Use deterministic fake component outputs for workflow checks, plus a few full cases with real components.

1. Apply approved text only to a local simulated ticket store. Give updates stable operation IDs and enforce uniqueness in storage.
2. Keep updates separate from approval pauses: work before an interrupt may execute again on resume.
3. Verify the essential cases below. Save outcomes and automate a few important checks.
4. Add workflow outcomes to RESULTS.md. Test documented setup and saved-artifact inference from a clean environment or checkout.
5. Record a brief CLI demonstration covering a supported request, an uncertain or missing-evidence case, and approval/rejection. Update the README with tested commands and limitations.

| Check | Required behavior |
| --- | --- |
| Pending or rejected reply | No update |
| Edited and approved draft | Apply the approved version |
| Restart while awaiting review | Resume the same pending ticket |
| Repeated approved operation | Exactly one mock update |
| Crash after update before checkpoint progress | Resume without a duplicate update |
| Model/retrieval failure | Clear failed/review state; no automatic approved update |

**Part 4 complete:** explicit approval, persistent resumption, retry protection, a working mock update, and a documented demonstration. Live integrations and an autonomous agent are unnecessary.

## GitHub commit checkpoints

Make commits when a meaningful piece works and you have checked it. The suggested checkpoints below follow the roadmap; they are not a quota. A day may need several commits or none. Commit completed work when it is ready rather than waiting for a particular date.

Use the suggested message only if it accurately describes what you implemented. Keep failures and fixes honest; do not manufacture dates, activity, or a polished history for recruiters. Clear changes, results, and explanations provide useful evidence of your progress.

| When ready | Check before committing | Suggested commit message |
| --- | --- | --- |
| Environment setup and roadmap | Dependency records exist and environment/cache paths are ignored | `chore: document environment and project roadmap` |
| Day 1: data inspection | Script prints schema, examples, and label counts | `feat: inspect support messages and select six intents` |
| Day 1: evaluation preparation | Split IDs are repeatable and duplicate groups stay together | `feat: add grouped evaluation splits and challenge cases` |
| Day 2: baseline | Saved pipeline reloads and validation results are recorded | `feat: train and evaluate TF-IDF intent baseline` |
| Day 3: encoder | Training, saved inference, and label mapping work | `feat: fine-tune encoder for support intent classification` |
| Day 3: comparison | Both models use the same evaluation protocol | `docs: compare intent models and explain prediction errors` |
| Day 4: uncertainty | Calibrator reloads and routing decisions are inspectable | `feat: calibrate intent scores and queue uncertain messages` |
| Day 5: Parts 1-2 | Frozen test results and actual run commands are saved | `docs: report held-out classification and routing results` |
| Day 6: knowledge base | Eight articles have stable IDs and clear policies | `feat: add fictional shop help articles` |
| Day 6: retrieval | Keyword and embedding methods have a development comparison | `feat: compare keyword and embedding retrieval` |
| Day 7: drafting | Drafts include inspectable evidence or insufficient-information status | `feat: draft cited support replies from retrieved policies` |
| Day 8: RAG evaluation | Held-out judgments and failures are recorded | `docs: evaluate retrieval and grounded reply quality` |
| Day 9: workflow | Branching, approval, and persistent resumption work | `feat: add persistent human approval workflow` |
| Day 10: retry protection | Pending/rejected drafts never apply and retries do not duplicate updates | `fix: prevent duplicate approved ticket updates` |
| Day 10: behavior checks | Repeatable approval, rejection, resume, and retry checks pass | `test: cover approval decisions and workflow recovery` |
| Day 10: release | Setup instructions and the demonstration were tested | `docs: add setup instructions and milestone demo` |

Use `fix` for an actual correction. If duplicate prevention is part of the initial implementation rather than a bug fix, use `feat: add idempotent mock ticket updates`. Separate documentation or test commits only when they are meaningful independent changes; combining related work is fine.

### How to make each commit

From the project folder, review your changes first:

```powershell
git status
git diff
```

Stage the specific files belonging to that change, inspect the staged diff, then commit. This example is for the initial setup checkpoint; adjust filenames to the files you actually created:

```powershell
git add .gitignore requirements.txt python-version.txt ROADMAP.md
git diff --cached
git commit -m "chore: document environment and project roadmap"
```

Check that no tokens, raw data, model weights, caches, or local ticket databases are staged. Small configuration files, appropriate split IDs, evaluation records, and source code can be committed. Avoid including unrelated edits just to get them into a commit.

The quoted text after `-m` is the commit message. A longer explanation is optional: use `git commit` without `-m` to open your editor, write a short first line, leave a blank line, and explain why the change was needed and how you checked it. Describe measured results only after running the experiment.

### Publish to GitHub

Local commits are not visible to recruiters until pushed to an accessible GitHub repository. Check the existing remote and branch first:

```powershell
git remote -v
git branch --show-current
```

Once the correct GitHub remote and branch upstream are configured, use `git push` after a checked checkpoint or at the end of a session. If no remote or upstream exists, configure it using your actual repository URL and branch; do not guess those values. This roadmap update does not create commits or push anything automatically.

Keep the README clear about what each milestone currently does. A small number of understandable commits with real evaluation evidence is more useful than a daily activity target.

## What to record

Keep one brief experiment log and one RESULTS.md with four milestone sections. For each meaningful experiment, record the question, data/model/configuration version, change, measured outcome, failures, and next step. Do not invent scores or business impact.

Save sufficient configuration and split IDs to interpret results. Keep weights, caches, credentials, and local databases outside ordinary Git commits; document how to recreate needed artifacts.

## How to start now

Continue your existing inspection script: load the data, show examples, count labels, then choose the six intents. Environment setup is already done. Focus only on Day 1 until its checks pass.

If ten sessions prove too tight, add a session for the current milestone. Keep the four-hour daily ceiling and the smaller scope.
