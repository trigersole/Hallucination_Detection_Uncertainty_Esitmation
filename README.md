# Controlled Verification for Hallucination Detection

**Recommended runner:** `bash scripts/run_assessment.sh --background`.
This supersedes the manual-wave submission instructions for new runs. It uses
one CPU per batch job, chat-formatted generation with token-limit retries,
per-question checkpoints, automatic scheduling, and an audited-label gate.
See "Single resumable cluster runner" at the end of this document. Historical
BLEURT-based metrics must not be treated as validated factual-correctness results.

This repository implements the pilot proposed in **“Beyond Generation
Confidence: Controlled Verification for Uncertainty-Based Hallucination
Detection.”** It keeps a causal language model frozen and extracts signals from:

- the original answer (confidence, consistency, and lexical-cluster entropy);
- three factual verification prompts;
- three structurally paired formatting-control prompts (not exact token-length matched);
- normalized hidden-state differences `verify - control` at selected layers.

It then trains calibrated linear detectors and compares generation-only,
P(True), verification-state, control-state, concatenation, and difference-state
feature sets.

The latest experiment is **concise generation + representation-only assessment**
(`--generation-prompt concise --evaluation-profile assessment-v3`). It includes
verification, formatting control, and a no-task condition at the shared
`Assessment:` cue, without label scoring. See the final section for launch steps.
The older score-based protocols below remain available for reproducibility.

## Install

Python 3.10+ and a CUDA-capable environment are recommended for 7B models.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Input data

Use JSON Lines with one object per question:

```json
{"id":"q1","question":"What is the capital of France?","answers":["Paris"]}
```

`answers` contains acceptable aliases. Automatic exact-match labeling is useful
for a pilot but must be manually audited for thesis results. Refusals and
ambiguous answers should be handled separately.

## Smoke-test extraction

```powershell
cverify extract --data data/example.jsonl --model Qwen/Qwen2.5-7B-Instruct --output runs/smoke --samples 2 --layers -1 -4 -8
```

For a real pilot, provide at least 20 questions with at least five correct and
five incorrect generated answers, then train the detectors:

```powershell
cverify extract --data data/pilot.jsonl --model Qwen/Qwen2.5-7B-Instruct --output runs/qwen_pilot --limit 1000 --samples 5 --layers -1 -4 -8
cverify train --run runs/qwen_pilot --output runs/qwen_pilot/results --bootstrap 1000
```

Add `--trust-remote-code` only for a model repository you trust. On limited
hardware, choose a smaller causal instruction model; the protocol is unchanged.

The extractor writes:

- `features.npz`: numeric features and labels;
- `metadata.jsonl`: prompts, answers, samples, probabilities, and correctness;
- `manifest.json`: model, layer, seed, and run settings.

The trainer uses a question-level stratified split, fits preprocessing only on
the training portion, calibrates on the development portion, and reports AUROC,
hallucination-positive AUPRC, Brier score, and bootstrap confidence intervals on
the held-out test set. It also reports error at 80% answer coverage and the
prespecified confident/consistent subgroup (defaults: mean token log-probability
at least `-0.5`, consistency at least `0.8`). Override those thresholds with
`--confidence-logprob` and `--consistency`; choose them before inspecting the
test set. Never select layers or prompts using the test set.

The comparison includes an original-answer hidden-state probe, making the key
incremental test `generation features + generation state` versus the full
controlled verifier. HaloScope, TSV, PRISM, and SSP are research baselines with
their own training protocols; integrate their published implementations against
the cached `metadata.jsonl` samples instead of relabeling or regenerating data.

## Important scope notes

- The included entropy groups normalized answer strings. It is a transparent
  **lexical-cluster proxy**, not full semantic entropy. Replace the clusterer
  with bidirectional NLI entailment for a publication-grade Semantic Entropy
  baseline.
- The verifier score is self-verification, not a ground-truth label.
- Hidden coordinates are model-specific; train separate probes per model.
- An equal-cost comparison should account for all generated and scored tokens.

## Full TruthfulQA experiment on Slurm

Export the 817-question generation split once, without the legacy HaloScope
dependency overlay:

```bash
cd /home/msai/siddhart022/Hallucination_Detection_Uncertainty_Esitmation
/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python \
  -m cverify.cli prepare-truthfulqa --output data/truthfulqa.jsonl
```

The included array launcher uses nine independent GPU shards of at most 100
questions, so no single extraction job approaches the six-hour cluster limit.
It uses the existing HaloScope BLEURT-20 checkpoint to label generated answers;
exact match is not appropriate for TruthfulQA's sentence-style answers. Because
this cluster permits only two submitted jobs at once, submit the shards in waves
and wait for each wave to finish before submitting the next:

```bash
sbatch --array=0-1%2 scripts/slurm_truthfulqa_array.sbatch
# Wait until both tasks finish successfully.
sbatch --array=2-3%2 scripts/slurm_truthfulqa_array.sbatch
# Wait, then repeat:
sbatch --array=4-5%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --array=6-7%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --array=8 scripts/slurm_truthfulqa_array.sbatch
```

After shard 8 finishes successfully, submit the CPU merge/training job:

```bash
sbatch scripts/slurm_truthfulqa_train.sbatch
```

Monitor all shards with:

```bash
squeue -u "$USER"
tail -f logs/cverify-tqa-ARRAY_JOB_0.out
```

The merge step rejects duplicate question IDs and mismatched model, layer,
sampling, or labeling settings. Final metrics are written to
`runs/truthfulqa/full/results/metrics.json`. Preserve all nine shard directories
until the merged file has been verified.

Do not queue the next wave or the training job early: pending tasks count toward
the two-job submission limit.

### Complete-sentence answer condition

Keep the original brief-answer run unchanged. After pulling the latest code,
export a separate run name and the new generation prompt in the shell from which
you submit the jobs:

```bash
export CVERIFY_RUN_NAME=truthfulqa_sentence
export CVERIFY_GENERATION_PROMPT=complete-sentence
```

Run shard 0 first as a pilot. Check `runs/truthfulqa_sentence/shard_0/metadata.jsonl`
for complete answers and truncation before committing to the other shards. Keep
the same settings for every shard; the merge command rejects mismatched prompt
profiles and maximum token limits.

```bash
sbatch --export=ALL --array=0 scripts/slurm_truthfulqa_array.sbatch
```

If the pilot looks good, submit the remaining shards in waves, waiting for each
wave to finish successfully before the next one:

```bash
sbatch --export=ALL --array=1 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=2-3%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=4-5%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=6-7%2 scripts/slurm_truthfulqa_array.sbatch
sbatch --export=ALL --array=8 scripts/slurm_truthfulqa_array.sbatch
```

After shard 8 finishes, merge and train on the prespecified seed-42 split:
512 training, 100 development, and 205 test questions.

```bash
sbatch --export=ALL scripts/slurm_truthfulqa_train.sbatch
```

Results go to `runs/truthfulqa_sentence/full/results_seed_42_val100_test25/`.
The original `runs/truthfulqa/` directory is untouched. The sentence prompt
encourages, but cannot guarantee, complete sentences. BLEURT is still an
imperfect correctness labeler: audit changed labels and ambiguous answers
before treating a detector score as factual hallucination performance.

## Corrected Yes/No protocol with layer 15

The corrected profile is `yes-no-v2`. It retains the three factual/formatting
task pairs, rewrites the imperative tasks as explicit questions, and asks both
conditions to respond `Yes` or `No`. The adapter scores **those same strings**,
not the legacy Correct/Incorrect labels. Hidden states are still read at the
last prompt token, before the response. This corrects the task/label mismatch;
it does not establish that formatting is a factually neutral control or that
subtraction isolates factual reasoning. No exact token-length matching is claimed.

The original profile remains available as `--evaluation-profile legacy` and is
the default for the original launcher. Corrected jobs use a separate run name:
`truthfulqa_sentence_yesno_l15`. Existing extraction directories cannot be
overwritten, and merging rejects different evaluation specifications or layers.
The manifest records the full evaluation template, instruction lists, scored
labels, extraction position, and resolved layer indices.

The corrected extraction includes `--layers -1 -4 -8 15`. These are indices into
Hugging Face's `hidden_states` tuple: index 0 is the embedding output; index 15
is the output after transformer block 15 (one-based), equivalent to zero-based
`model.layers[14]`. For the 32-block Llama-2-7B model, the resolved tuple indices
are `[32, 29, 25, 15]`. Zero-based block 15 would instead require tuple index 16.

On the cluster, first synchronize these code changes using your normal workflow.
Then, from the repository directory:

```bash
mkdir -p logs
export CVERIFY_RUN_NAME=truthfulqa_sentence_yesno_l15
sbatch --export=ALL --array=0 scripts/slurm_truthfulqa_corrected_array.sbatch
```

The wrapper defaults to the same Llama-2-7B-Chat model, complete-sentence answer
prompt, sampling settings, BLEURT labeler, and per-shard generation seeds as the
original sentence run. It **regenerates** answers; it does not reuse cached
answer token IDs, which the original run did not save. It checks all regenerated
answers, sampled answers, and binary correctness labels against
`runs/truthfulqa_sentence/shard_N/` and fails on a mismatch. This is required to
avoid silently changing the evaluated answers or stratified splits. Keep the
same model/checkpoint, environment and shard size as the original run. If your
original shards are elsewhere, set `CVERIFY_REFERENCE_RUN` to the matching
original shard before each submission. Do not point a later shard at shard 0.

After the first shard succeeds, inspect its manifest and metadata. For example:

```bash
cat runs/truthfulqa_sentence_yesno_l15/shard_0/manifest.json
head -n 2 runs/truthfulqa_sentence_yesno_l15/shard_0/metadata.jsonl
```

Submit the remaining shards in waves, waiting for each wave to finish before
submitting the next (pending jobs count toward the two-job limit):

```bash
sbatch --export=ALL --array=1 scripts/slurm_truthfulqa_corrected_array.sbatch
# Wait for completion before each following submission:
sbatch --export=ALL --array=2-3%2 scripts/slurm_truthfulqa_corrected_array.sbatch
sbatch --export=ALL --array=4-5%2 scripts/slurm_truthfulqa_corrected_array.sbatch
sbatch --export=ALL --array=6-7%2 scripts/slurm_truthfulqa_corrected_array.sbatch
sbatch --export=ALL --array=8 scripts/slurm_truthfulqa_corrected_array.sbatch
```

Once all nine shards succeed:

```bash
sbatch --export=ALL scripts/slurm_truthfulqa_corrected_train.sbatch
```

This merges once and trains three prespecified layer conditions on the same
seed-42 split (512 train, 100 development, 205 test):

- `results_seed_42_val100_test25_original_layers`: `-1 -4 -8` only.
- `results_seed_42_val100_test25_layer15`: `15` only.
- `results_seed_42_val100_test25_all_layers`: all four layers.

All results are under `runs/truthfulqa_sentence_yesno_l15/full/`. Compare the
original-layer condition against the original sentence experiment to assess the
prompt correction. Compare the three corrected conditions to assess adding layer
15. Do not attribute a change from the old three-layer run to the prompt alone
when comparing against a corrected four-layer run, or choose layers on test data.
Further split seeds can be trained from these features with `cverify train
--layers ... --seed ...`, using separate result directories. Regularization
tuning and learning-curve experiments remain a subsequent step.

## Concise generation with verification, control, and none (assessment-v3)

This is a fresh generation experiment, separate from the sentence/Yes-No runs.
The first answer and four additional samples all use:

```text
Answer the question concisely.
Question: {question}
Answer:
```

There is one instruction per assessment condition (not a three-paraphrase
ensemble). All three conditions use the same original greedy answer:

```text
Question: {question}
Proposed answer: {answer}
Task: Assess the factual correctness of the proposed answer for this question.
Assessment:
```

```text
Question: {question}
Proposed answer: {answer}
Task: Assess the formatting clarity of the proposed answer.
Assessment:
```

```text
Question: {question}
Proposed answer: {answer}
Assessment:
```

The last condition is named `none`: it omits the entire Task line, not the
question/answer or shared cue. `Assessment:` still provides assessment framing;
this is not a completely unprompted representation. Each condition runs one
forward pass over its prompt. Read hidden states at the final token of
`Assessment:`; no assessment continuation is generated and no Yes/No or
Correct/Incorrect likelihoods are scored.

The feature file includes generation behavior/state, `verify_state`,
`control_state`, `none_state`, verification/control concatenation, and normalized
state differences `verify - control` (`difference_state`), `verify - none`, and
`control - none`. Each layer is unit-normalized before subtraction. No
`verification_scores` array or verification-probability metadata is written.
BLEURT correctness labels remain necessary as the detector training/evaluation
target; removing response-label scoring does not remove correctness labeling.

The trainer automatically uses 14 representation-based feature sets: three
generation baselines; the three assessment conditions; concatenation and
difference for each of the three condition pairs; and generation behavior/state
combined separately with the verify-control and verify-none differences. It
does not train P(True), generation-plus-verification-score, or the old score-based
full-controlled detector. Metric configuration records feature dimensions.

The new launcher retains Llama-2-7B-Chat, five answers, sampling temperature 0.7,
32 maximum new tokens, BLEURT threshold 0.5, and layers `-1 -4 -8 15`. The
`15` index uses the convention described above. The manifest records the exact
generation/evaluation templates, none template, layer order, and extraction
position. It separately counts assessment-prompt input tokens; this is still
not a complete generation/replay/BLEURT computational-cost measurement.

After synchronizing the updated code to the cluster, launch the first shard:

```bash
cd /home/msai/siddhart022/Hallucination_Detection_Uncertainty_Esitmation
mkdir -p logs
export CVERIFY_RUN_NAME=truthfulqa_concise_assessment_l15
sbatch --export=ALL --array=0 scripts/slurm_truthfulqa_assessment_array.sbatch
```

The wrapper clears any inherited `CVERIFY_REFERENCE_RUN`: new generation prompts
must not be required to reproduce the old complete-sentence answers. Check the
new shard's answers, truncation, and manifest before the remaining waves:

```bash
cat runs/truthfulqa_concise_assessment_l15/shard_0/manifest.json
head -n 2 runs/truthfulqa_concise_assessment_l15/shard_0/metadata.jsonl
```

Wait for each wave to succeed before the next submission; the two-job limit
includes pending tasks:

```bash
sbatch --export=ALL --array=1 scripts/slurm_truthfulqa_assessment_array.sbatch
# Wait before each subsequent wave:
sbatch --export=ALL --array=2-3%2 scripts/slurm_truthfulqa_assessment_array.sbatch
sbatch --export=ALL --array=4-5%2 scripts/slurm_truthfulqa_assessment_array.sbatch
sbatch --export=ALL --array=6-7%2 scripts/slurm_truthfulqa_assessment_array.sbatch
sbatch --export=ALL --array=8 scripts/slurm_truthfulqa_assessment_array.sbatch
```

After all nine shards succeed, merge and train:

```bash
sbatch --export=ALL scripts/slurm_truthfulqa_assessment_train.sbatch
```

Results are under `runs/truthfulqa_concise_assessment_l15/full/`, with separate
seed-42 result directories for original layers, layer 15 alone, and all layers.
All condition comparisons within a layer setting use the same generated answer
and split. The generation prompt and evaluation protocol both differ from the
old sentence runs, so cross-run performance changes cannot be attributed to a
single cause. Original extraction directories are never overwritten.

For a small local-compatible smoke extraction (choose an installed model):

```bash
python -m cverify.cli extract --data data/example.jsonl --model MODEL \
  --output runs/assessment_smoke --generation-prompt concise \
  --evaluation-profile assessment-v3 --layers -1 -4 -8 15 --samples 2
```

## Single resumable cluster runner (recommended)

Run from the existing cluster repository on `codex/concise-assessment-l15`:

```bash
git pull --ff-only
bash scripts/run_assessment.sh --background
```

Do not submit the controller itself with `sbatch`. The lightweight controller
runs on the login host and submits its workers to Slurm. The same file is used
as the workers' batch script. It submits at most two jobs at once, counting other
jobs belonging to your user. GPU extraction workers request one GPU, **one CPU**,
32 GB RAM and 05:50:00. Finalization/training workers request one CPU and no GPU.
Thread-count environment variables are set to one. All legacy `.sbatch` files
also now request one CPU.

No pip installation or manual exports are required for the configured cluster.
The script uses `/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python`, sets
`PYTHONPATH` to this checkout's `src`, and avoids the old BLEURT dependency
overlay. Override `CVERIFY_PYTHON` if that executable is elsewhere.

The new default run directory is:

```text
runs/truthfulqa_concise_chat_audit_l15/
```

This deliberately does not inherit `CVERIFY_RUN_NAME`, `CVERIFY_MAX_NEW_TOKENS`,
`CVERIFY_SHARD_SIZE`, or reference-run exports left over from earlier experiments.
It preserves all previous runs. Settings are frozen in `workflow_config.json`:

- Llama-2-7B-Chat with tokenizer chat formatting; no raw completion prompting.
- Exact concise generation instruction and verification/control/none tasks from
  `assessment-v3`; same original answer in all three assessment conditions.
- `Assessment:` is an **assistant prefill after the user turn**, not a cue
  trapped inside the user instruction. No assessment continuation is generated.
- If an older official Llama-2 cache has no chat template, use the explicit
  single-user `[INST]` format from [Meta's reference implementation](https://github.com/meta-llama/llama/blob/main/llama/generation.py).
  Unknown models without templates fail rather than silently using raw text.
- Five answers per question; one greedy plus four samples at temperature 0.7.
- Initial generation ceiling **256 new tokens**. A length-stopped answer is
  regenerated at 512, then 1,024 tokens using the same prompt and seed. Text is
  not silently cut or edited. EOS at the budget boundary is considered completion.
- Metadata records final token IDs/counts, stopping reason, retry history, and
  suspected extra `Question:` continuations. A higher ceiling does not guarantee
  completeness; unresolved outputs remain flagged for exclusion.
- Generation states are read at the last non-EOS answer token. Assessment states
  are read at the last token of the shared assistant `Assessment:` prefix.
- Layers `-1 -4 -8 15`; layer 15 means hidden-state tuple index 15, after the 15th
  transformer block. Seeds are based on global question position, not shard-local
  position, so repartitioning does not redefine the random seed for a question.
- 25 questions per shard (33 shards for 817 questions). Smaller shards accommodate
  longer generation under the six-hour limit; the controller handles all waves.

If needed, set overrides **before the first invocation**, for example:

```bash
export CVERIFY_PIPELINE_RUN=truthfulqa_concise_chat_audit_l15
export CVERIFY_PIPELINE_INITIAL_TOKENS=256
export CVERIFY_PIPELINE_MAX_TOKENS=1024
export CVERIFY_PIPELINE_SHARD_SIZE=25
# Optional: use an existing input file rather than exporting TruthfulQA:
# export CVERIFY_PIPELINE_DATA=/absolute/path/to/truthfulqa.jsonl
bash scripts/run_assessment.sh --background
```

If no input path is supplied, the runner downloads/exports TruthfulQA once into
the new run directory, including correct and incorrect references. A supplied
existing JSONL file can omit incorrect references. The controller freezes the
dataset hash, source-code hash, and settings; workers reject subsequent changes.
Model commit, library versions and exact chat template are also checked across
extraction checkpoints and shards. Do not pull/edit code while a run is active.

### Resume and monitor

```bash
tail -f logs/assessment-controller.log
bash scripts/run_assessment.sh --status
squeue -u "$USER"
```

Start or resume with the **same command**:

```bash
bash scripts/run_assessment.sh --background
```

There is no need to track wave numbers or manually queue merge/training.
An OS lock rejects duplicate controllers. The controller persists submitted job
IDs; it does not duplicate active jobs and reserves slots for jobs not yet visible
in `squeue`. It uses `sacct` to distinguish terminal jobs, retries incomplete
workers up to three times per invocation, and stops on unknown scheduler state
rather than risking duplicate work. After fixing an external problem, rerun the
same command. Existing per-question checkpoints and completed training layer
conditions are reused. `nohup` handles ordinary terminal disconnection, but host
reboots or site policies can still terminate the controller; restart it to resume.

Each question is saved atomically under `shard_N/checkpoints/`. Interrupted output
files are rebuilt from these checkpoints. A killed question can be recomputed;
earlier committed questions are retained. Old non-checkpointed runs are not
silently adopted. Use a new `CVERIFY_PIPELINE_RUN` if changing code/settings.

### Correctness review is required before metrics

BLEURT similarity is no longer the new runner's correctness target. Extraction
writes `is_correct: null` and `y_error=-1`. Training refuses these unreviewed
targets. After extraction, one CPU job merges the shards and writes:

```text
runs/truthfulqa_concise_chat_audit_l15/review.jsonl
```

Each line contains the question, original generated answer, references, quality
flags, and fields to complete:

```json
{"judgment": "incorrect", "reviewer": "reviewer-name-or-audited-label-source", "notes": "Answer contradicts the reference on the date."}
```

Edit those fields **in each existing record**, preserving IDs, question/answer
text, and `sample_hash`. Allowed judgments are `correct`, `incorrect`, `ambiguous`,
`refusal`, and `exclude`. The reference lists are evidence to inspect, not an
infallible automatic judge. Review the whole answer, including extra claims.
For thesis results, use documented human adjudication of uncertain cases; the
software cannot establish factual correctness from an unvalidated proxy.

The controller exits with status 2 and `needs_review` when judgments are pending;
that is an intentional review gate, not an extraction failure. It will not
overwrite your edits or fabricate metrics. `review.summary.json` reports counts.
After completing the review, run the same background command again. It skips
GPU extraction, applies the audited targets, and submits CPU training automatically.

Answers marked ambiguous/refusal/exclude are omitted from the binary evaluation.
Questions with empty answers, non-EOS generations (including sampled answers),
or extra generated questions are excluded even if their correctness was reviewed;
these would otherwise contaminate generation-consistency features. Exclusions
and label provenance are recorded. Metrics describe this retained population,
not the excluded questions. With exclusions, the 25% test and development
fraction remain fixed but split counts can differ from 512/100/205.

Results are versioned by review-file hash under `audited/<hash>/`, for original
layers, layer 15, and all four layers. Editing labels creates a new result version
rather than replacing an earlier reviewed result. No new GPU generation is needed
for label corrections. This still uses lexical entropy, not full semantic entropy;
regularization/learning-curve comparisons remain separate research work.

### Recover a completed extraction after the finalization-lock fix

The runner now explicitly requests one node and one task, and calls
`srun --ntasks=1 --cpus-per-task=1`. Another live finalizer holding the advisory
lock is treated as existing work, not a reason to crash or submit duplicates.
Do not delete a live lock file: the operating system releases its lock when the
holder exits, and unlinking it can allow concurrent writers.

For an existing run with all extraction shards complete, pull the update and use:

```bash
git pull --ff-only
bash scripts/run_assessment.sh --background --finalize-existing
```

This mode reuses the completed shards and permits only merge/review/training.
It validates the original dataset, shard sizes, completion fingerprints and
extraction settings, preserves the original `workflow_config.json`, and records
the new orchestration code hash in a separate `finalize_config_<hash>.json`.
It cannot submit GPU extraction. This avoids weakening the normal code-version
check or forcing another extraction merely to install a scheduler fix. Use this
same command again after filling the review file or after an interruption.

```bash
bash scripts/run_assessment.sh --status
tail -f logs/assessment-controller.log
```

An unchanged review file with pending judgments still returns `needs_review`.
The lock fix does not assign correctness labels or erase the history of failed
jobs in Slurm accounting.

### Disk-quota recovery and reviewed feature views

Reviewed runs now store small `labels.npz` files containing retained source row
indices and correctness labels. Their manifests reference the original merged
`full/features.npz`, with SHA-256 checks of source features, metadata and manifest.
Training reads and subsets those features in memory; it does not write another
large feature archive for every review version. Keep the source `full/` directory:
these reviewed datasets are indexed views, not independent copies. Existing
physical archives remain supported by the training CLI.

Atomic JSON/NPZ writers remove incomplete temporary files after an exception and
preserve any previous completed target. Quota/capacity failures in finalization
are reported as `storage_blocked` with exit code 75; the controller stops automatic
retries for that failure until the user resumes after making space available.

An older failed job may have left a large **incomplete** `features.npz.tmp` under
an `audited/<review-hash>/` directory. After confirming that no worker is writing
it, remove only that failed temporary copy, not the extraction or review data:

```bash
RUN=runs/truthfulqa_concise_chat_audit_l15
find "$RUN/audited" -type f -name 'features.npz.tmp' -print -delete
git pull --ff-only
bash scripts/run_assessment.sh --background --finalize-existing
```

Do not remove `full/features.npz`, completed shard files/checkpoints, or
`review.jsonl`. If more space is needed, inspect the account's quota and storage
locations rather than deleting source data blindly. A run still needs a small
amount of free space for code updates, reviewed metadata, labels, logs and metrics.

### Training representation subspaces

Compare the top 10 covariance eigenvectors of none, verification and control using
the exact saved seed-42 training rows (no development or test rows):

```bash
mkdir -p logs
sbatch scripts/compare_subspaces.sbatch
```

This is a standalone one-CPU analysis with no GPU request or feature extraction.
It finds the completed audited run from `finalize_status.json`. To select another
workflow run, pass its directory as the first batch-script argument. The Python
entry point also accepts `--metrics` for another saved training split and `--k`.

The audited directory receives `subspace_top10_train.json`, containing separate
results for every extracted layer, the original three layers and all layers.
Each condition is centered on its own training mean in the shared raw hidden
coordinate system, without coordinate standardization. Eigenspaces are computed
using the sample Gram matrix, avoiding a large feature covariance matrix. Outputs
include explained variance, the relative eigengap at the component cutoff,
absolute 10-by-10 eigenvector cosine matrices, principal angles and subspace
overlap (mean squared cosine of principal angles). Overlap is 1 for identical
subspaces and 0 for orthogonal subspaces; it is invariant to basis signs and
rotations. Small cutoff eigengaps indicate sensitivity to the arbitrary top-10
boundary. High overlap describes dominant variance geometry and does not prove
equal correctness information or uncertainty. Centering removes mean shifts;
these results do not describe condition mean differences. The source features
and original metrics remain unchanged.
