# CANDOR Annotation Split

Tools for carving the CANDOR conversations into **two subsets that are sufficiently distinct
according to the human ground-truth annotations**. The idea: systematically search the
annotation columns (and their combinations), and for each candidate find a low/high partition
of conversations that differ by at least a configurable margin.

## Contents

| File | Purpose |
| --- | --- |
| `annotation_split.py` | Search annotation columns/combinations for two well-separated partitions. |
| `sanity_check_partitions.py` | Recompute a saved partition's margins from scratch, to validate it. |
| `browse_splits.html` | Self-contained browser UI to interactively filter/sort/inspect a results JSON. |
| `fetch_annotations.py` | Download **only** the annotations from HuggingFace (no audio) and aggregate them to conversation level. |
| `BetterUp CANDOR Corpus Data Dictionary - survey.csv` | The BetterUp survey schema: every raw survey column, its question text, response scale, and notes. |

All scripts are **pure Python standard library** (`fetch_annotations.py` additionally needs
`huggingface_hub`). No numpy / pandas / scipy required.

---

## Example run and sanity check

Run the split using at most two features/question to determine a split. Note that increasing this `--max-combo` value quickly requires a lot of memory. Increasing it to e.g. 5 already requires more than 32GB of memory due to the high number of combinations.

```
python annotation_split.py --metrics all_surveys_raw.csv --group-by convo_id --max-combo 2 --margin 1.5 --out-json try01_full.json --write-partitions
```

This command creates a `paritions` subdirectory with separate `.csv` files for each combination. To run a sanity check of the created paritions, you can run `sanity_check_partitions.py`:

```
python sanity_check_partitions.py partitions/affect+best_arousal__assignment.csv --metrics all_surveys_raw.csv
```

This is output something like:

```
== affect+best_arousal__assignment.csv ==
features: affect+best_arousal   id: convo_id
sizes: low=113 high=113 mid=225 unmatched=0
composite (high vs low): cohens_d=5.8279 auc=1.0000 p=0.00e+00 z_gap=1.2509
per-column (high vs low):
    column                       high_mean  low_mean  raw_diff  cohens_d     auc
    affect                          8.4071    6.1416    2.2655    3.3905  0.9906
    best_arousal                    8.4690    5.7301    2.7389    3.9692  0.9977
label_mix: low={'': 113} high={'': 113} mid={'': 225}
```

## Data model

CANDOR ground-truth annotations come from the post-conversation **survey** each participant
filled in. There are several views of it:

1. **Raw, per-respondent** — one `survey.csv` per conversation on the Hub (or a concatenated
   `all_surveys_raw.csv`), with ~240 columns (see the data dictionary) and one row per
   respondent — so **two rows per `convo_id`**, keyed uniquely by (`convo_id`, `user_id`).
   Examples: `how_enjoyable`, `i_like_you`, `in_common`, the 8 shared-reality items
   `*_sr1..sr8`, the `responsive_1..3` items, personality (`my_bfi_*`), demographics, etc.
   Likert/multiple-choice answers are numeric-coded; free-text and nominal fields are strings.

2. **Curated, conversation-level** — the in-repo file
   [`../src/data/survey_metrics_conversation_level.csv`](../src/data/survey_metrics_conversation_level.csv),
   one row per conversation, with a small set of aggregated numeric metrics plus **derived**
   fields that are *not* in the raw surveys:

   | Column | Meaning |
   | --- | --- |
   | `convo_id` | Conversation id (key). |
   | `how_enjoyable`, `shared_reality`, `responsive`, `conversationalist`, `i_like_you`, `in_common` | Aggregated numeric survey annotations. |
   | `paper_pcs_proxy` | Derived "positive conversation score" proxy (0–1). |
   | `label` | Curated success category: `HSC` (high), `MSC` (medium), `LSC` (low), `UNSCORED`. |
   | `dyad_gender` | e.g. `female-male`. |
   | `<col>_n_raters` | Number of raters behind each metric (reliability). |

   Distribution: 1510 `MSC`, 91 `HSC`, 35 `LSC`, 20 `UNSCORED` (1656 rows).

3. **Aggregated by participant** — e.g. `surveys_agg_by_participant.csv`, one row per
   `user_id` (no `label`), where each per-conversation item appears as a `<base>_mean` /
   `<base>_std` pair aggregated over that participant's conversations, alongside
   participant-level fields (`age`, personality `my_*`, psychometric `*_score`, and
   categorical `sex` / `race` / `politics` / `employ` / `edu`).

`annotation_split.py` runs on **any** of these views. It:
- **auto-detects the id column** (`convo_id` / `user_id` / …) and, when the primary id repeats
  (the raw per-respondent file), extends it to a unique composite key (`convo_id/user_id`) so
  output ids stay unambiguous. Pass `--id-col a,b` for an explicit composite. The row-based
  scoring does not depend on id uniqueness.
- **classifies `<base>_mean`/`<base>_std` columns via their base name** in the data dictionary,
  so coded categoricals (`end_you_mean`, `politics`, …) are excluded from either the raw or the
  aggregated files.
- **tolerates missing cells**: rows are not dropped for incomplete data; each candidate subset
  uses exactly the rows that have all of its columns (each conversation/respondent counts once
  per candidate).

For the participant file, add `--exclude-suffixes _std` to search the mean levels only.
`fetch_annotations.py` produces a raw-derived conversation-level equivalent for the full Hub
dataset. Note the unit of analysis differs per view: conversations (curated), respondents (raw),
or participants (aggregated) — but `--group-by convo_id` collapses the raw per-respondent file
to conversation level on the fly (averaging the respondents; `--group-agg` picks the reducer),
so you can search conversation-level partitions directly from `all_surveys_raw.csv`.

---

## `annotation_split.py` — the partition search

### What it does

For every candidate set of annotation columns, it builds a one-dimensional separation score,
splits the conversations into a **low** and a **high** partition, measures how different the
two partitions are, and keeps only the splits whose separation clears a configurable margin.

### Logic, step by step

0. **Select feature columns dynamically.** The candidate columns are **not** hardcoded. If
   `--feature-cols` is omitted, `auto_select_features` chooses them from the input by combining
   two signals:
   - **the data** — a column is usable only if it is present, mostly float-parseable
     (`--min-numeric-frac`, default 0.8) and varies across conversations; and
   - **the data dictionary** (`--data-dict`, default the bundled BetterUp CSV) — each survey
     column is classified from its `type`/`selector`/`choices` as `numeric` (Likert matrices,
     sliders, constant-sum %, computed scores, ordinal multiple-choice), `categorical`
     (Yes/No, `sex`, `race`, `politics`, …), `free_text` (open text boxes) or `meta` (ids).

   Only `numeric` columns are kept; `categorical` / `free_text` / `meta` are dropped **even
   when they are numerically coded** in the data (e.g. `sex` = 1/2) — which naive
   float-detection would wrongly treat as a feature. Ids, the label column, `*_n_raters`,
   `--exclude-cols`, and a few known metadata columns are always excluded. Columns **absent
   from the dictionary** (e.g. the derived `paper_pcs_proxy`) are kept in the default lenient
   mode and dropped under `--data-dict-strict`, which restricts the search to documented
   survey items. `--data-dict none` disables the dictionary (data-only detection); inspect the
   outcome with `--list-features`. On the curated file this reproduces the original 7 columns.

1. **Load & filter.** Read the conversation-level CSV. Drop rows whose `label` is in
   `--exclude-labels` (default `UNSCORED`). Optionally require each column to have at least
   `--min-raters` raters (using the `<col>_n_raters` columns when present). A row is kept for
   a candidate only if it has a numeric value for every column in that candidate.

2. **Enumerate candidates.** All column subsets of size `--min-combo .. --max-combo`
   (e.g. 7 columns → 7 singles, 63 subsets up to 3-way). This is the "systematic" search.

3. **Score.** Within the retained rows, z-score each constituent column (subtract mean,
   divide by std) and combine into a single score = **mean of the z-scores**. Z-scoring puts
   columns on different scales (e.g. `conversationalist` 0–100 vs `i_like_you` 1–7) on equal
   footing; the mean is meaningful because the default annotations all point "higher = better".

4. **Split with a guaranteed gap.** Sort conversations by score. Take the bottom
   `--group-frac` as the **LOW** partition and the top `--group-frac` as the **HIGH**
   partition; discard the middle band. This guarantees a separation between the groups and
   keeps them balanced. `--group-frac 0.25` = bottom vs top quartile; `0.5` = median split.

5. **Measure separation.** Between the two partitions, on the composite score and on every
   constituent column:
   - **Cohen's *d*** — standardized mean difference (pooled std).
   - **AUC** — common-language effect size = P(random high value > random low value), from a
     tie-corrected Mann-Whitney U; 0.5 = no separation, 1.0 = perfect.
   - **Mann-Whitney p-value** — via normal approximation (tie-corrected).
   - **z_gap** — gap between the bands in z units; `> 0` means the two partitions don't even
     overlap on the score.
   - **min_column_cohens_d** — the weakest constituent column's *d* (the meaningful
     discriminator for combinations).

6. **Qualify by margin.** Keep a split iff `--margin-metric` ≥ `--margin`:
   - `cohens_d` (default) — separation on the composite score.
   - `auc` — `2·|AUC − 0.5|` (0…1).
   - `raw_gap` — the z_gap.

   Optionally also require, with `--require-all-columns`, that **every** constituent column
   separates by ≥ `--margin` (so both partitions are genuinely distinct on all chosen
   annotations), and `--alpha` for Mann-Whitney significance.

7. **Rank & output.** Sort survivors by the margin metric; print the top `--top-k`. With
   `--out-json`, write all qualifying splits (with per-column stats and label mix); with
   `--write-partitions`, write, per split, `<features>__low.txt`, `<features>__high.txt`, and
   `<features>__assignment.csv` (`convo_id,partition`).

### Why the effect sizes look large — and why they're still meaningful

Splitting at the extremes of a score and then measuring separation on that same score is
optimistic **by construction**, so absolute *d* values are not "real" effect sizes. But
because `--group-frac` is fixed across all candidates, the number is a **fair ranking
signal**: a larger value means that annotation (or combination) separates the population more
cleanly at that fraction. The honest external checks are:
- `label_mix` — does the split line up with the curated HSC/LSC labels?
- `min_column_cohens_d` — for combos, do *all* chosen annotations actually separate?

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `--metrics` | `../src/data/survey_metrics_conversation_level.csv` | Input CSV. |
| `--id-col` / `--label-col` | *auto* / `label` | Key and category columns. Id auto-detected (comma-separated for a composite key; auto-extended to stay unique on the raw per-respondent file). |
| `--feature-cols` | *auto* | Columns to search over. If omitted, selected dynamically (step 0). |
| `--data-dict` | bundled BetterUp CSV | Dictionary used to keep only ordinal/numeric items (`none` disables). |
| `--data-dict-strict` | off | Restrict to columns documented as numeric in the dictionary. |
| `--exclude-cols` | — | Extra columns to exclude from auto-selection. |
| `--exclude-suffixes` | — | Column-name suffixes to drop, e.g. `_std` (keep only `_mean` levels). |
| `--min-numeric-frac` | `0.8` | Fraction of values that must parse as float for a column to count as numeric. |
| `--list-features` | off | Print the auto-selected columns (and why others were skipped), then exit. |
| `--exclude-labels` | `UNSCORED` | Labels to drop. |
| `--min-raters` | `0` | Minimum rater count per column. |
| `--group-by` | — | Aggregate rows sharing these column(s) into one unit first, e.g. `convo_id` to collapse the raw per-respondent survey to conversation level. The group column(s) become the id. |
| `--group-agg` | `mean` | How to combine each feature across a group: `mean` \| `median` \| `min` \| `max`. |
| `--min-combo` / `--max-combo` | `1` / `1` | Subset sizes to enumerate. |
| `--group-frac` | `0.25` | Fraction taken from each extreme. |
| `--min-group-size` | `20` | Minimum conversations per partition. |
| `--margin` | `1.0` | Required separation. |
| `--margin-metric` | `cohens_d` | `cohens_d` \| `auc` \| `raw_gap`. |
| `--require-all-columns` | off | Every constituent column must also clear the margin. |
| `--alpha` | none | Require Mann-Whitney p < alpha. |
| `--all-combos` | off | Keep every tried combination (each tagged `qualified`), not only those clearing the margin — for validating the search. |
| `--top-k` | `25` | How many to print (text mode). |
| `--json` | off | Emit results as JSON to stdout instead of the text table (diagnostics stay on stderr). |
| `--out-json` | — | Also write the results JSON to a file. |
| `--json-ids` | off | Include the low/high/mid partition member ids in JSON output (omitted by default). |
| `--write-partitions` / `--out-dir` | — | Write per-split id lists + a full low/high/mid assignment CSV to a directory. |

Both `--json` and `--out-json` produce the same object: `{params, n_candidates, n_qualified,
n_results, results}`, where each result carries `features`, `score`, `per_column`,
`min_column_cohens_d`, `qualified`, `label_mix` (per-partition label counts for `low`/`high`/
`mid`), and — with `--json-ids` — the full membership `low_ids` / `high_ids` / `mid_ids`. The
three id lists partition every conversation the candidate could score (`low + high + mid =
n_used`), so you can validate exactly which conversations landed in each group. `mid` is the
discarded middle band between the two extremes. `--write-partitions` additionally writes, per
split, `<features>__low.txt`, `<features>__high.txt`, and a `<features>__assignment.csv` with a
`low`/`high`/`mid` label per id.

Example output of `annotation_split.py --max-combo 2 --margin 1.5 --json --json-ids`
(abbreviated — `results` holds one object per qualifying split, ranked by the margin metric;
id lists truncated here with `...`):

```json
{
  "params": {
    "metrics": ".../survey_metrics_conversation_level.csv",
    "id_cols": ["convo_id"],
    "features": ["how_enjoyable", "shared_reality", "responsive", "conversationalist",
                 "i_like_you", "in_common", "paper_pcs_proxy"],
    "n_features": 7,
    "group_by": null,
    "group_agg": null,
    "min_combo": 1,
    "max_combo": 2,
    "group_frac": 0.25,
    "min_group_size": 20,
    "margin": 1.5,
    "margin_metric": "cohens_d",
    "require_all_columns": false,
    "alpha": null,
    "all_combos": false
  },
  "n_candidates": 28,
  "n_qualified": 28,
  "n_results": 28,
  "results": [
    {
      "features": ["in_common"],
      "n_used": 1636,
      "group_size": 409,
      "score": { "cohens_d": 5.5961, "auc": 1.0, "p_value": 0.0, "z_gap": 1.282 },
      "per_column": {
        "in_common": { "high_mean": 7.78, "low_mean": 3.8093, "raw_diff": 3.9707,
                       "cohens_d": 5.5961, "auc": 1.0 }
      },
      "min_column_cohens_d": 5.5961,
      "qualified": true,
      "low_ids":  ["0278950b-a7e0-4e15-8a2b-1629ff1b17ba", "9e11ca71-...", "..."],
      "high_ids": ["a52ee968-49e1-47c8-b02a-dc5fa78ffe16", "a57b07c4-...", "..."],
      "mid_ids":  ["1507fb8f-a797-4911-a695-15e9856cb173", "1e35a2d3-...", "..."],
      "label_mix": {
        "low":  { "HSC": 2,  "LSC": 31, "MSC": 376 },
        "high": { "HSC": 69, "LSC": 1,  "MSC": 339 },
        "mid":  { "HSC": 20, "LSC": 3,  "MSC": 795 }
      }
    }
  ]
}
```

Note how `label_mix` corroborates the split: the low `in_common` partition is enriched for
`LSC` (31 vs 1) and the high partition for `HSC` (69 vs 2), while the discarded `mid` band is
overwhelmingly `MSC`. The `low_ids` / `high_ids` / `mid_ids` lists (present with `--json-ids`)
account for every scored conversation, so you can check exactly which conversations landed in
each group for each tried combination.

### Examples

```bash
# Inspect which columns are auto-selected (and why others are skipped)
python annotation_split.py --list-features

# Strongest single-annotation splits (bottom vs top quartile), Cohen's d >= 1.5
python annotation_split.py --max-combo 1 --margin 1.5

# Up to 3-way combos; both partitions must differ on every constituent column;
# require significance; write the winning partitions to disk.
python annotation_split.py --max-combo 3 --require-all-columns \
    --margin 1.0 --alpha 0.01 --write-partitions --out-json out/qualifying.json

# Balanced halves (median split) ranked by common-language AUC
python annotation_split.py --group-frac 0.5 --margin-metric auc --margin 0.9

# Machine-readable JSON on stdout (with per-combination low/high/mid membership), piped to jq
python annotation_split.py --max-combo 2 --margin 1.5 --json --json-ids | jq '.results[0]'

# Validate the full search: keep every tried combination (qualified flag) with its assignments
python annotation_split.py --max-combo 2 --all-combos --json --json-ids --out-json out/all.json

# Participant-aggregated file: id auto-detected as user_id, search mean levels only
python annotation_split.py --metrics surveys_agg_by_participant.csv \
    --exclude-suffixes _std --max-combo 2 --margin 2.0

# Raw per-respondent file, unit = respondent: id auto-extends to composite convo_id/user_id
python annotation_split.py --metrics all_surveys_raw.csv --max-combo 2 --margin 1.5

# Raw per-respondent file, unit = conversation: aggregate the 2 respondents per convo first
# (mean by default; --group-agg median|min|max). Id becomes convo_id.
python annotation_split.py --metrics all_surveys_raw.csv --group-by convo_id \
    --max-combo 2 --margin 1.5

# ...same, writing the conversation-level partitions to disk
python annotation_split.py --metrics all_surveys_raw.csv --group-by convo_id \
    --group-agg median --max-combo 2 --margin 1.5 --write-partitions
```

Reading a result line:

```
in_common   n=1636 grp=409 d=5.596 auc=1.000 p=0.0e+00 z_gap=1.282 min_col_d=5.596
```

= splitting on `in_common` over 1636 usable conversations, 409 per partition; the high/low
groups are ~5.6 pooled-SDs apart, never overlap (`z_gap > 0`), and — from the JSON
`label_mix` — the low group is enriched for `LSC` and the high group for `HSC`.

---

## `sanity_check_partitions.py` — validate a saved partition

An independent recomputation of a split's margins, so you can trust the search output. It reads
a `<features>__assignment.csv` (written by `--write-partitions`), looks the ids up in the source
data, and recomputes the separation between the `low` and `high` partitions **from scratch**,
reusing the same stats functions as `annotation_split.py`. If the numbers match the search, the
saved partition is faithful.

It reports, high vs low:
- the **composite** Cohen's d / AUC / Mann-Whitney p / z-gap (mean of z-scores over the split's
  features, z-scored across all assigned rows = `low + high + mid`);
- **per-column** Cohen's d / AUC / raw mean difference for each feature;
- optional **`--extra-cols`** — how well the *same* partition separates other annotations (e.g.
  does an `in_common` split also separate `paper_pcs_proxy`?);
- the **label mix** of each partition, and a `direction_ok` check (the `high` partition's scores
  should exceed `low`; flagged `INVERTED` otherwise).

Feature columns come from the assignment filename (`a+b__assignment.csv` → `a`, `b`) unless
`--feature-cols` is given. Point `--metrics` at the same source you searched; if several source
rows share an id (the raw per-respondent survey split with `--group-by convo_id`), pass the same
`--group-agg` so they are aggregated identically.

```bash
# Recompute margins for one saved split (default conversation-level source)
python sanity_check_partitions.py partitions/in_common__assignment.csv

# Check every split in a directory; also report separation on the success proxy
python sanity_check_partitions.py partitions/ --extra-cols paper_pcs_proxy

# A split built from the raw per-respondent survey aggregated to conversation level
python sanity_check_partitions.py partitions/how_enjoyable__assignment.csv \
    --metrics all_surveys_raw.csv --group-agg mean --json
```

Example (matches what the search reported for the `in_common` split — d=5.60):

```
== in_common__assignment.csv ==
features: in_common   id: convo_id
sizes: low=409 high=409 mid=818 unmatched=0
composite (high vs low): cohens_d=5.5961 auc=1.0000 p=0.00e+00 z_gap=1.282
per-column (high vs low):
    column                       high_mean  low_mean  raw_diff  cohens_d     auc
    in_common                       7.7800    3.8093    3.9707    5.5961  1.0000
    paper_pcs_proxy                 0.8428    0.6615    0.1814    1.7243  0.8962  (extra)
label_mix: low={'HSC': 2, 'LSC': 31, 'MSC': 376} high={'HSC': 69, 'LSC': 1, 'MSC': 339} ...
```

**Caveat — the `mid` band matters for an exact match.** The composite z-scoring baseline is the
full set of assigned rows (`low + high + mid`), matching how the search z-scored over all
*usable* rows. So the recomputation reproduces the reported composite `cohens_d` / `z_gap`
exactly only when the assignment CSV includes the `mid` rows (current `--write-partitions`
output does). An older assignment file with just `low`/`high` would z-score over a smaller set
and the composite numbers would drift slightly — the printed `sizes` (`mid=…`) and `unmatched`
count make that visible. The **per-column** stats (Cohen's d, AUC, raw diff) only use the `low`
and `high` values and are unaffected either way.

---

## `browse_splits.html` — interactive viewer

A single self-contained HTML file (no dependencies, no network, no build step) for exploring a
results JSON written by `annotation_split.py --out-json` (e.g. `try01_full.json`, which can hold
tens of thousands of splits). Everything runs locally in the browser.

Open it and load the JSON one of two ways:

```bash
# 1) Just open the file and pick/drag the JSON in the page (works from file://):
xdg-open browse_splits.html            # or double-click it

# 2) Serve the folder so it auto-loads a co-located file (fetch is blocked on file://):
python3 -m http.server 8000
#   then visit  http://localhost:8000/browse_splits.html
#   (auto-loads try01_full.json; use ?src=other.json for a different file)
```

It shows the run `params` up top and a sortable, filterable, paginated table of every split:
- **filter** by feature substring, number of features, min composite Cohen's d, min per-column d,
  max p-value, and qualified-only;
- **sort** by any score column (click a header, or use the dropdown) — default is the run's
  margin metric;
- **click a split** to expand its per-column separation (with a Cohen's d bar), the `low`/`high`/
  `mid` label mix, the full score, and member ids if the JSON was written with `--json-ids`;
- **click a feature chip** to filter to splits containing it.

Large files (15 MB / ~17k splits) load and browse smoothly since only the current page of rows
is rendered.

---

## `fetch_annotations.py` — annotations only, no audio

To run the search on the **full** CANDOR dataset from the Hub
(`JSALT2026-Conv-AI-Simulator/CANDOR`) without downloading the audio.

### Why it's cheap

The dataset is one folder per conversation:

```
{convo_id}/processed/{convo_id}.mp3          <- audio: the bulk of the dataset
{convo_id}/transcription/transcript_*.csv    <- transcripts
{convo_id}/survey.csv                         <- the survey (annotations), a few KB
```

`snapshot_download(..., allow_patterns=["*/survey.csv"])` lists the repo (a metadata-only
call) and downloads **only** the survey files — a few MB total instead of the full
multi-hundred-GB audio corpus. It is parallel, resumable and cached.

### Logic

1. **(Optional) list.** `--list-only` prints every non-audio file so you can spot a
   repo-level metadata file before downloading anything.
2. **Download.** `snapshot_download` with `allow_patterns=["*/survey.csv"]` into `--raw-dir`.
3. **Aggregate.** Each `survey.csv` has one row per respondent. Collapse to one row per
   conversation (default `--agg mean`; also `first` / `min` / `max` / `both`), keyed by the
   survey's own `convo_id`. Numeric columns are coerced to float; empty cells ignored.
4. **Write.** A wide `candor_annotations_conversation_level.csv`, columns ordered by how
   well-populated they are, with a printed suggestion of well-populated `--feature-cols`.

### Caveats

- The curated `label` (HSC/LSC/MSC) and `paper_pcs_proxy` are **derived** and are **not** in
  the raw surveys. Use `--list-only` to check for a repo-level metadata file if you need them
  for the full dataset. This script does not try to re-derive `paper_pcs_proxy`.
- Averaging the two respondents turns directional items (e.g. `i_like_you` = "how much *I*
  liked my partner") into a symmetric conversation-level quantity (mutual liking). Use
  `--agg first` or `--agg both` if that is not what you want.
- Requires `huggingface_hub` (present in the `jsalt_env` conda env). If the dataset is gated,
  authenticate first: `huggingface-cli login` or set `HF_TOKEN`.

### Usage

```bash
conda activate jsalt_env
huggingface-cli login                 # only if the dataset is gated

python fetch_annotations.py --list-only          # inspect what's there
python fetch_annotations.py                       # fetch + aggregate -> CSV
python fetch_annotations.py --limit 50            # prototype on 50 conversations first
```

---

## End-to-end

```bash
# 1. Get just the annotations from the Hub (no audio)
conda activate jsalt_env
python fetch_annotations.py            # -> candor_annotations_conversation_level.csv

# 2. Search for two sufficiently-distinct partitions and write them out
python annotation_split.py \
    --metrics candor_annotations_conversation_level.csv \
    --feature-cols how_enjoyable,i_like_you,in_common,conversationalist \
    --max-combo 3 --margin 1.0 --write-partitions
```

## Output artifacts (git-ignored candidates)

Running with persistence produces `partitions/` and any `out/` JSON, plus
`fetch_annotations.py` writes `candor_surveys_raw/` and the aggregated CSV. These are
generated data — consider adding them to `.gitignore` rather than committing.
