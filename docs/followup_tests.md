# Follow-up tests on completed reviewed runs

These are optional extensions after the two-dataset matched study. They do not
replace or relabel the completed experiments. All new outputs are under each
current audited directory's `followup/` subdirectory. Both datasets and their
all-layer seed42 metrics must already be complete. No original extractor,
training or cached combined-analysis source files are changed by this addition.

## Questions and limits

| Test | What it measures | What it cannot establish |
|---|---|---|
| Question-only versus full context | Whether answer-conditioned states add predictive information beyond an answer-free representation on identical splits | Unique causal answer information: token positions and context change when answers are omitted |
| Reviewed same-question alternatives | Whether a frozen detector assigns greater error probability to an incorrect than a correct answer to the identical question | A pure factual mechanism: answer style, lexical cues and length can still differ |
| Frozen TruthfulQA to TriviaQA | Performance when no TriviaQA data enter detector fitting or calibration | Fresh prospective validation: the target dataset/results were already inspected |
| Layer/verification/subtraction sweep and geometry | Where discrimination and representation differences occur | A new confirmatory winner after many comparisons, or proof that a PCA direction represents hallucination |

Previously chosen primary matched results stay fixed. Report these extensions
as follow-up/exploratory work, including null and adverse effects. Do not keep
adding configurations until a favourable result appears.

## 1. Pull and identify current paths

Run in the existing cluster checkout, using the existing branch:

```bash
git pull --ff-only
mkdir -p logs
export CVERIFY_PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
PYTHONPATH=src "$CVERIFY_PYTHON" -m cverify.followup paths
```

The command resolves `finalize_status.json` for both workflows, rather than
guessing review-hash directory names. Default workflows are
`runs/truthfulqa_concise_chat_audit_l15` and
`runs/triviaqa_validation500_replication_reviewed`. All commands accept explicit
`--truth PATH --trivia PATH` to select different completed workflows or audited
directories. Do not point them at raw, unreviewed features.

## 2. CPU-only existing-feature analyses

```bash
sbatch scripts/run_followup.sbatch cpu
```

This runs, in sequence:

1. **Frozen transfer:** fit scalers and balanced probes on TruthfulQA seed42
   train, Platt calibration on TruthfulQA dev, reproduce saved source-test AUROC
   and Brier, and evaluate all 476 included reviewed TriviaQA answers. The
   original source test labels and all TriviaQA labels stay out of fitting.
   First-pass, none, verification, control, control+none and fixed ensemble are
   reported. Source/target representation compatibility and normalized question
   overlap (including excluded source answers) are checked.
2. **Detailed TriviaQA detection:** each individual layer, original three and
   all four layers, across split seeds42-46. First-pass, verification-only,
   control-only, none, verify-control, verify-none, control-none, control+none
   and fixed ensemble. Subtractions operate on L2-normalised hidden vectors,
   not probabilities. Train/dev/test separation and original all-layer baseline
   reproduction are enforced. Checkpoints resume completed layer/seed fits.
3. **Existing geometry:** top10 train-only covariance components, first/second
   pass comparisons, none/verify/control subspaces and cross-condition transfer.
   It invokes the same modules used for TruthfulQA, with TriviaQA's saved splits.

Outputs under TriviaQA's `followup/`: `frozen_transfer/summary.txt`,
`detailed_detection/summary.txt`, `geometry/summary.txt`,
`subspaces/subspace_top10_train.json`, and
`condition_transfer/share_transfer_results.json`. Detailed JSON and prediction
files accompany summaries. `cpu_status.json` becomes complete only after all
stages finish. Resubmit the command if a time-limited job stops; completed
fitting/geometry checkpoints are retained. Frozen probability caches are reused.

## 3. Question-only extraction and comparison

```bash
sbatch --gres=gpu:1 scripts/run_followup.sbatch question
```

This processes both datasets, in sequence, with the same pinned model, tokenizer,
runtime, layers and per-layer L2 normalization. It performs forwards only:
**no answers are generated or changed**. Four answer-free features are saved:

- `first`: the original concise-generation prompt before answer generation;
- `none`: `Question: {question}` followed by the assistant `Assessment:` cue;
- `verify`/`control`: the same factual/formatting task instructions and cue,
  omitting the entire proposed-answer field.

The task's wording stays fixed, even though no answer is supplied. This is an
answer-omission ablation. No references, generated answers or labels enter any
of these prompts. The `first` feature is the final generation-input context
position, whereas the original first-pass feature is after the generated answer.
The second-pass ablations retain the assistant cue, but context lengths differ.

After that job completes:

```bash
sbatch scripts/run_followup.sbatch question-score
```

Fit each answer-free and full-context probe independently with identical original
question splits, train-only scalers/probes, dev-only calibration, C=0.1 and five
split seeds. Report full-minus-question-only AUROC/Brier with paired question
bootstrap intervals. Look for consistent positive AUROC and negative Brier
deltas, not simply a high full-context AUROC. Question-only can predict difficulty
of a single observed answer; it is not automatically a hallucination detector
that distinguishes alternate answers to the same question.

Each dataset writes `followup/question_detection/summary.txt` and `report.json`.
GPU extraction uses per-question checkpoints and validates provenance on resume.

## 4. Same-question reviewed substitutions

First export candidates without loading a model:

```bash
sbatch scripts/run_followup.sbatch pairs-export
```

This writes `followup/reviewed_pairs.json` for each dataset, using **only the
original seed42 held-out questions** (161 TruthfulQA,119 TriviaQA). Original
reviewed answers and available references are candidate text, not automatically
validated pair labels. Missing incorrect answers are deliberately left blank.

For every question, review/edit `correct_answer` and `incorrect_answer`; fill
`reviewer` and evidence in `notes`. Keep the question and question ID unchanged.
Use natural answers of comparable length, style and specificity. Avoid labels
like "correct answer", overtly absurd distractors or systematic short-reference
versus long-generated-answer contrasts. Incorrect alternatives must be plausible
and demonstrably factually wrong; correct alternatives must answer the question
without materially false additions. To omit an unusable question, record a
nonempty `skip_reason`. Do not choose or edit pairs based on detector scores.
Report all skips, their reasons and the selection procedure; the code requires
at least ten pairs but small samples will yield imprecise evidence.

Then:

```bash
sbatch --gres=gpu:1 scripts/run_followup.sbatch pairs-extract
```

After extraction completes:

```bash
sbatch scripts/run_followup.sbatch pairs-score
```

Both answer versions use the same none/verify/control assessment prompts and
cue. Original seed42 native probes are fitted on original training questions,
calibrated on original dev questions and checked against original test metrics.
They remain frozen when scoring substituted answers. No alternative answers
are used for fitting or calibration. The fixed native ensemble is also scored.

The report includes mean `p_error(incorrect)-p_error(correct)`, within-question
ordering accuracy (ties count one half), and intervals resampling whole question
pairs. An answer-free detector gives identical scores to both alternatives, so
its pair-order reference is 0.5. Improvement above 0.5 indicates discrimination
within these reviewed pairs, subject to answer-selection and lexical confounds.
These are controlled answer substitutions, not a causal intervention on model
beliefs or a generation accuracy experiment.

Each dataset writes `followup/pair_detection/summary.txt` and predictions. Editing
a pair after extraction invalidates its fingerprint; use a separate output for
changed experiments rather than mixing checkpoints.
Use `--output-name followup_v2` consistently on all relevant commands to select
a separate follow-up directory if a reviewed pair protocol or extraction code changes.

## Resources and sharing

Every job requests one CPU,32GB RAM and5h50. Only `question` and `pairs-extract`
need a GPU; request one with the shown `--gres` flag and use your usual GPU
partition if required. Run dependent stages only after their inputs finish.
Start at most the CPU job and one GPU job together to respect the existing
two-job limit. Other user jobs also occupy those slots.

Inspect `logs/followup-JOBID.out`/`.err`. Use `followup paths` for exact summary
locations. Share both question-only and pair-test summaries and TriviaQA's
frozen-transfer/detailed/geometry summaries. Large feature archives do not need
to be pasted. Models/labels/splits from the completed thesis runs are preserved.

Local verification uses synthetic fixtures and a fake model adapter for forward
extraction. Real Llama runtime/GPU execution must be verified on the cluster;
unit tests do not claim these research results have already been obtained.
