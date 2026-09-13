# Judging Spoken Human and AI Conversations

Measuring how successful a conversation is remains difficult, even for humans judging spoken
dialogue. We evaluate state-of-the-art LLMs as pointwise and pairwise judges of conversational
success on CANDOR, finding pointwise scoring correlates moderately with human ratings, while
pairwise comparison suffers from long transcripts and positional bias. Since this leaves judge
scores incomparable across models, we propose a small anchor set and a calibration function
that calibrates any judge onto a shared, interpretable scale. We further release the VoiceArena
Goal Dataset (VA), 200 task-oriented human-AI and human-agent conversations with pairwise
annotations, revealing a substantial gap between current judges and human-level discrimination.
Using VA, we test whether CANDOR-fitted calibration transfers to human-AI conversations,
finding it brings judges onto a shared scale despite never observing VA during fitting.

## Setup

Different judge models need different, often mutually incompatible `transformers`
versions/forks, so dependencies are split into per-model extras in
[pyproject.toml](pyproject.toml). Install the base package plus the extra for the model(s)
you want to run, e.g.:

```
# Qwen2.5-Omni, Qwen3-Omni, Qwen3.5
pip install -e ".[qwen]"

# Phi-4-multimodal
pip install -e ".[phi]"

# Qwen3.6-27B
pip install -e ".[qwen36]"
```

## Usage

### Pointwise judging ([scripts/01_run_pointwise.sh](scripts/01_run_pointwise.sh))

Score conversations with a pointwise judge, then evaluate (Spearman, Krippendorff, AUC)
against ground truth:

```
bash scripts/01_run_pointwise.sh
```

### Pairwise judging ([scripts/02_run_pairwise.sh](scripts/02_run_pairwise.sh))

Pairwise (best-worst-scaling) judging. Needs a batches file (one JSON line per batch:
`batch_id`, `ordering_idx`, `convo_ids`, `n_convs`, `gold_ranking`) that this repo does not
build for you:

```
bash scripts/02_run_pairwise.sh
```

### Calibration on CANDOR ([scripts/03_calibrate_candor.sh](scripts/03_calibrate_candor.sh))

Fit calibration for one or more judges on CANDOR and reproduce the anchor-size/method
comparison table. Needs `judge_scores_dataset.jsonl` (raw per-judge CANDOR scores) and
`pointwise_dataset.jsonl` (ground truth), built from your own
[run_judge.py](src/run_judge.py) runs. Uses the 32-conversation anchor set in
[candor_anchor_ids_32.txt](src/calibration/candor_anchor_ids_32.txt):

```
bash scripts/03_calibrate_candor.sh
```

### Calibration transfer to VoiceArena ([scripts/04_calibrate_va.sh](scripts/04_calibrate_va.sh))

Apply the CANDOR-fit calibration to VoiceArena scores and check cross-judge agreement
before/after. Needs `voicearena_success_scores.jsonl` (per-judge VoiceArena scores), same
caveat:

```
bash scripts/04_calibrate_va.sh
```

Each script is a short example; edit the flags to use your own model/prompt/judges.
