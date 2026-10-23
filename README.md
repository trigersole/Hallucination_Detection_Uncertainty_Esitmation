# Controlled Verification for Hallucination Detection

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
