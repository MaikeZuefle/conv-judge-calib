#!/usr/bin/env python3
"""Systematically search CANDOR annotation columns for two well-separated partitions.

The goal of this project is to carve the CANDOR conversations into two subsets that are
*sufficiently distinct* according to the human ground-truth annotations. This script does
that search systematically: it enumerates single annotation columns and their combinations,
builds a 1-D separation score for each candidate, splits the conversations into a low group
and a high group, and keeps only the splits whose two partitions differ by at least a
configurable margin (effect size).

Input
-----
A conversation-level annotations CSV keyed by an id column. The default is the in-repo
ground-truth file produced for the LLM-judge work:

    ../src/data/survey_metrics_conversation_level.csv

whose numeric annotation columns are:
    how_enjoyable, shared_reality, responsive, conversationalist, i_like_you,
    in_common, paper_pcs_proxy
each paired with a `<col>_n_raters` reliability column, plus a categorical `label`
(HSC / MSC / LSC / UNSCORED) and `dyad_gender`.

The script is not tied to that particular file: point `--metrics` at any wider survey
export that has an id column and the requested numeric feature columns.

Method
------
1. Filter rows (drop unwanted `--exclude-labels`; optionally require `--min-raters`).
2. For every subset of `--feature-cols` of size 1..`--max-combo`:
     * z-score each constituent column across the retained rows,
     * combine into one score = mean of the z-scores (sign-aligned; all default
       annotations point "higher = better", so a plain mean is meaningful).
3. Split by score: bottom `--group-frac` = LOW partition, top `--group-frac` = HIGH
   partition. The middle band is discarded, which guarantees a gap between the groups and
   keeps the two partitions balanced in size.
4. Measure how different the two partitions are and keep the split iff the chosen
   `--margin-metric` clears `--margin` (and, optionally, every constituent column also
   clears it via `--require-all-columns`, and Mann-Whitney p < `--alpha`).
5. Rank the survivors and (optionally) write out the partition membership.

Because the group fraction is fixed across all candidates, the effect size becomes a fair
ranking signal: a larger value means the annotation (or combination) separates the
population more cleanly at that fraction.

Everything is pure standard library so it runs without numpy/pandas/scipy.

Examples
--------
    # Single-column splits, default file, keep splits with Cohen's d >= 1.5
    python annotation_split.py --max-combo 1 --margin 1.5

    # Up to 3-way combinations, both partitions must differ on every constituent column,
    # require significance, and write the winning partitions to disk.
    python annotation_split.py --max-combo 3 --require-all-columns \
        --margin 1.0 --alpha 0.01 --write-partitions

    # Balanced halves (median split) instead of extreme quartiles
    python annotation_split.py --group-frac 0.5 --margin 0.8

    # Machine-readable results on stdout (with partition membership), e.g. piped to jq
    python annotation_split.py --max-combo 2 --margin 1.5 --json --json-ids | jq '.results[0]'
"""

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_METRICS = _HERE.parent / "src" / "data" / "survey_metrics_conversation_level.csv"
DEFAULT_DATADICT = _HERE / "BetterUp CANDOR Corpus Data Dictionary - survey.csv"

# Feature columns are selected dynamically from the input data (see auto_select_features),
# using the data dictionary to keep only ordinal/numeric survey metrics. This list is only a
# last-resort fallback if auto-detection finds nothing.
FALLBACK_FEATURES = [
    "how_enjoyable",
    "shared_reality",
    "responsive",
    "conversationalist",
    "i_like_you",
    "in_common",
    "paper_pcs_proxy",
]

# Never treat these as partition features even when they are numeric.
ALWAYS_EXCLUDE = {"short_id", "hf_recording_path", "dyad_gender", "n_respondents", "respondent",
                  "n_conversations"}

# Id columns tried (in order) when --id-col is not given.
COMMON_ID_COLS = ["convo_id", "user_id", "id", "short_id"]

# Aggregation suffixes stripped when looking a column up in the data dictionary, so that
# e.g. `how_enjoyable_mean` / `how_enjoyable_std` are classified from base `how_enjoyable`.
_AGG_SUFFIXES = ("_mean", "_std")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _is_unique(rows, cols):
    seen = set()
    for r in rows:
        key = tuple(r.get(c, "") for c in cols)
        if key in seen:
            return False
        seen.add(key)
    return True


def resolve_id_cols(header, rows, id_arg):
    """Return the list of columns forming the row identifier.

    An explicit `--id-col` may name several comma-separated columns (a composite key). When
    omitted, the first of COMMON_ID_COLS present is used; if that is not unique across rows
    (e.g. `convo_id` in the raw per-respondent survey, two rows per conversation), it is
    extended with `user_id`/`partner_id` until the combination is unique, so output ids stay
    unambiguous. Computation itself does not rely on id uniqueness.
    """
    if id_arg:
        cols = [c.strip() for c in id_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in header]
        if missing:
            sys.exit(f"[error] id column(s) {missing} not found. Columns: {header}")
        return cols

    primary = next((c for c in COMMON_ID_COLS if c in header), None)
    if primary is None:
        sys.exit(f"[error] could not auto-detect an id column among {COMMON_ID_COLS}. "
                 f"Pass --id-col explicitly. Columns: {header}")
    cols = [primary]
    if not _is_unique(rows, cols):
        for extra in ("user_id", "partner_id"):
            if extra in header and extra not in cols:
                cols.append(extra)
                if _is_unique(rows, cols):
                    break
    return cols


def load_rows(path, id_arg):
    """Read the CSV and resolve the id column(s). Returns (rows, id_cols)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[error] no rows in {path}")
    id_cols = resolve_id_cols(list(rows[0]), rows, id_arg)
    note = " (composite, to disambiguate duplicate ids)" if len(id_cols) > 1 else ""
    print(f"[info] using id column(s) {id_cols}{note}", file=sys.stderr)
    return rows, id_cols


def _to_float(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def prepare(rows, id_cols, features, label_col, exclude_labels, min_raters):
    """Return a list of {id, label, values:{col:float}} holding whatever features each row has.

    Missing/insufficiently-rated cells are simply omitted from a row's `values`; rows are NOT
    dropped for incomplete data (that would be listwise deletion across all features, which
    discards almost everything on wide, sparse files). Each candidate subset later restricts
    itself to the rows that actually have all of its columns (see evaluate_subset). A row is
    kept if it has at least one usable feature value.
    """
    header = set(rows[0])
    for col in features:
        if col not in header:
            sys.exit(f"[error] feature column {col!r} not found. Columns: {sorted(header)}")

    rater_cols = {col: f"{col}_n_raters" for col in features if f"{col}_n_raters" in header}
    exclude = set(exclude_labels or [])

    kept = []
    for row in rows:
        if label_col in row and row[label_col] in exclude:
            continue
        values = {}
        for col in features:
            v = _to_float(row[col])
            if v is None:
                continue
            if min_raters and col in rater_cols:
                n = _to_float(row[rater_cols[col]])
                if n is None or n < min_raters:
                    continue
            values[col] = v
        if values:
            rid = "/".join(row.get(c, "") for c in id_cols)
            kept.append({"id": rid, "label": row.get(label_col, ""), "values": values})
    return kept


def _reduce(vals, how):
    if how == "mean":
        return sum(vals) / len(vals)
    if how == "min":
        return min(vals)
    if how == "max":
        return max(vals)
    if how == "median":
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0
    raise ValueError(how)


def aggregate_rows(rows, group_cols, feature_cols, label_col, how):
    """Collapse rows sharing the same `group_cols` key into one row per group.

    Each feature is reduced across the group's rows with `how` (missing cells ignored); the
    group key columns are preserved, the label is taken from the first labelled member, and a
    `n_respondents` count is added. Every feature column is always present (empty when the
    group had no values for it) so downstream header checks stay consistent.
    """
    groups = {}
    order = []
    for row in rows:
        key = tuple(row.get(c, "") for c in group_cols)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    out = []
    for key in order:
        members = groups[key]
        agg = dict(zip(group_cols, key))
        for col in feature_cols:
            vals = [v for v in (_to_float(m.get(col)) for m in members) if v is not None]
            agg[col] = str(_reduce(vals, how)) if vals else ""
        if label_col:
            labels = [m.get(label_col, "") for m in members if m.get(label_col, "")]
            agg[label_col] = labels[0] if labels else ""
        agg["n_respondents"] = str(len(members))
        out.append(agg)
    return out


# --------------------------------------------------------------------------- #
# Data dictionary & dynamic feature selection
# --------------------------------------------------------------------------- #
# Identity / metadata survey columns that are never partition features.
_DICT_META = {"user_id", "partner_id", "convo_id", "date", "survey_duration_in_seconds", "time_zone"}
# Words that mark an ordinal (ordered) scale in a multiple-choice item whose response levels
# are stored as text rather than as leading digits (e.g. Likert "Strongly disagree..agree").
_ORDINAL_WORDS = (
    "disagree", "agree", "not at all", "extremely", "characteristic",
    "hardly ever", "some of the time", "often", "never", "strongly", "slightly",
)


def load_data_dictionary(path):
    """Parse the BetterUp data dictionary into {column: {'types', 'selectors', 'choices'}}.

    A column may appear on several rows (re-worded questions, matrix sub-items); we union the
    `type`/`selector` values and keep the first non-empty `choices` string.
    """
    info = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                col = (row.get("column") or "").strip()
                if not col or col == "column":
                    continue
                e = info.setdefault(col, {"types": set(), "selectors": set(), "choices": ""})
                e["types"].add((row.get("type") or "").strip().lower())
                e["selectors"].add((row.get("selector") or "").strip().lower())
                choices = (row.get("choices") or "").strip()
                if choices and not e["choices"]:
                    e["choices"] = choices
    except OSError as exc:
        sys.exit(f"[error] could not read data dictionary {path}: {exc}")
    return info


def classify_column(col, entry):
    """Classify a dictionary column as 'numeric' (ordinal/interval, usable as a feature),
    'categorical', 'free_text', or 'meta'. Unknown columns return 'unknown'."""
    if col in _DICT_META:
        return "meta"
    types, selectors = entry["types"], entry["selectors"]
    choices = entry["choices"].lower()

    # Free-text answers (an open text box), e.g. *_memory_text, critical_positive.
    if "text entry" in types and "text box" in selectors:
        return "free_text"
    # Numeric response formats: Likert matrices, sliders, constant-sum %, computed scores.
    if types & {"matrix table", "slider", "constant sum", "descriptive block"}:
        return "numeric"
    if "text entry" in types and "slider" in selectors:  # numeric slider entry (age, sleep, conv_length)
        return "numeric"
    if "multiple choice" in types:
        # Ordinal if the answer levels form a numeric or worded scale; else a nominal category
        # (Yes/No, sex, race, employment...) that must not be partitioned on even if int-coded.
        if any(ch.isdigit() for ch in choices) or any(w in choices for w in _ORDINAL_WORDS):
            return "numeric"
        return "categorical"
    return "unknown"


def classify_with_suffix(col, dict_info):
    """Classify `col`, first directly then via its base name after stripping an aggregation
    suffix (`_mean`/`_std`). Returns a classification string or None if not in the dictionary."""
    if col in dict_info:
        return classify_column(col, dict_info[col])
    for suf in _AGG_SUFFIXES:
        if col.endswith(suf):
            base = col[: -len(suf)]
            if base in dict_info:
                return classify_column(base, dict_info[base])
            break
    return None


def is_numeric_in_data(rows, col, min_frac):
    """True if `col` is present, mostly float-parseable, and varies across rows."""
    present = [row[col].strip() for row in rows if col in row and row[col] and row[col].strip()]
    if len(present) < 2:
        return False
    floats = [_to_float(v) for v in present]
    n_ok = sum(1 for v in floats if v is not None)
    if n_ok < min_frac * len(present):
        return False
    return len({v for v in floats if v is not None}) > 1


def auto_select_features(rows, id_cols, label_col, dict_info, strict, extra_exclude, min_frac,
                         exclude_suffixes=()):
    """Dynamically choose partition-feature columns from the input data.

    A column qualifies when it is numeric-and-varying in the data, is not an id/label/rater
    or explicitly excluded column, and the data dictionary does not mark it (or its base name,
    after stripping an aggregation suffix) as categorical / free-text / metadata. Columns
    absent from the dictionary are kept in the default (lenient) mode -- so derived metrics
    such as `paper_pcs_proxy` survive -- but dropped under `strict`, which restricts the
    search to columns documented as ordinal/numeric survey items. Columns ending in any of
    `exclude_suffixes` (e.g. `_std`) are dropped up front.
    """
    header = list(rows[0].keys())
    excluded = set(extra_exclude) | ALWAYS_EXCLUDE | set(id_cols) | {label_col}
    exclude_suffixes = tuple(s for s in exclude_suffixes if s)
    selected, skipped = [], {}
    for col in header:
        if col in excluded or col.endswith("_n_raters"):
            continue
        if exclude_suffixes and col.endswith(exclude_suffixes):
            skipped[col] = "excluded-suffix"
            continue
        if not is_numeric_in_data(rows, col, min_frac):
            continue
        cls = classify_with_suffix(col, dict_info) if dict_info else None
        if cls in ("categorical", "free_text", "meta"):
            skipped[col] = cls
            continue
        if strict and cls != "numeric":  # not documented as an ordinal/numeric survey item
            skipped[col] = "not-in-dictionary" if cls is None else cls
            continue
        selected.append(col)
    return selected, skipped


# --------------------------------------------------------------------------- #
# Statistics (pure python)
# --------------------------------------------------------------------------- #
def mean(xs):
    return sum(xs) / len(xs)


def std(xs, ddof=0):
    if len(xs) - ddof <= 0:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def zscores(records, col):
    """Return z-scores for `col`, aligned positionally with `records`.

    Positional (not keyed by id) so inputs with duplicate ids -- e.g. the raw per-respondent
    survey, which has two rows per convo_id -- are handled correctly.
    """
    xs = [r["values"][col] for r in records]
    m, s = mean(xs), std(xs)
    s = s or 1.0  # guard against a constant column
    return [(x - m) / s for x in xs]


def cohens_d(a, b):
    """Standardized mean difference (high group `a` minus low group `b`), pooled std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = std(a, ddof=1) ** 2, std(b, ddof=1) ** 2
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U with tie-corrected normal approximation.

    Returns (auc, p) where auc = P(random high-group value > random low-group value),
    the common-language effect size, in [0, 1]; 0.5 means no separation.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.5, 1.0
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    # Rank with ties averaged.
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u_a = rank_a - na * (na + 1) / 2.0
    auc = u_a / (na * nb)

    mu = na * nb / 2.0
    # Tie correction for the variance.
    n = na + nb
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    var = na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return auc, 1.0
    z = (u_a - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return auc, max(0.0, min(1.0, p))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Split evaluation
# --------------------------------------------------------------------------- #
def evaluate_subset(records, subset, group_frac, min_group_size):
    """Score, split and measure one feature subset. Returns a result dict or None.

    Only rows that have every column in `subset` are used, so wide/sparse inputs still yield
    as many conversations as each candidate individually supports.
    """
    usable = [r for r in records if all(col in r["values"] for col in subset)]
    if len(usable) < 2:
        return None

    # Composite score = mean of z-scores across the subset (z-scored over the usable rows).
    zcols = {col: zscores(usable, col) for col in subset}
    scored = []
    for i, r in enumerate(usable):
        score = mean([zcols[col][i] for col in subset])
        scored.append((score, r))
    scored.sort(key=lambda t: t[0])

    n = len(scored)
    k = int(round(group_frac * n))
    if k < max(1, min_group_size) or 2 * k > n:
        return None

    low = [r for _, r in scored[:k]]
    high = [r for _, r in scored[-k:]]
    mid = [r for _, r in scored[k:-k]]  # discarded middle band (kept for validation)

    low_scores = [s for s, _ in scored[:k]]
    high_scores = [s for s, _ in scored[-k:]]

    d_score = cohens_d(high_scores, low_scores)
    auc_score, p_score = mann_whitney(high_scores, low_scores)
    z_gap = min(high_scores) - max(low_scores)  # separation in z units (>0 => disjoint bands)

    # Per-column separation between the two partitions (raw annotation units + effect size).
    per_column = {}
    for col in subset:
        hi = [r["values"][col] for r in high]
        lo = [r["values"][col] for r in low]
        per_column[col] = {
            "high_mean": round(mean(hi), 4),
            "low_mean": round(mean(lo), 4),
            "raw_diff": round(mean(hi) - mean(lo), 4),
            "cohens_d": round(cohens_d(hi, lo), 4),
            "auc": round(mann_whitney(hi, lo)[0], 4),
        }

    return {
        "features": list(subset),
        "n_used": n,
        "group_size": k,
        "score": {
            "cohens_d": round(d_score, 4),
            "auc": round(auc_score, 4),
            "p_value": p_score,
            "z_gap": round(z_gap, 4),
        },
        "per_column": per_column,
        "min_column_cohens_d": round(min(c["cohens_d"] for c in per_column.values()), 4),
        "low_ids": [r["id"] for r in low],
        "high_ids": [r["id"] for r in high],
        "mid_ids": [r["id"] for r in mid],
        "label_mix": {
            "low": _label_counts(low),
            "high": _label_counts(high),
            "mid": _label_counts(mid),
        },
    }


def _label_counts(records):
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return dict(sorted(counts.items()))


def margin_value(result, metric):
    if metric == "cohens_d":
        return abs(result["score"]["cohens_d"])
    if metric == "auc":
        return abs(result["score"]["auc"] - 0.5) * 2.0  # 0..1, 1 == perfect separation
    if metric == "raw_gap":
        return result["score"]["z_gap"]
    raise ValueError(metric)


def qualifies(result, args):
    if margin_value(result, args.margin_metric) < args.margin:
        return False
    if args.require_all_columns and result["min_column_cohens_d"] < args.margin:
        return False
    if args.alpha is not None and result["score"]["p_value"] >= args.alpha:
        return False
    return True


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_partitions(result, out_dir, id_cols):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "+".join(result["features"])

    (out_dir / f"{tag}__low.txt").write_text("\n".join(result["low_ids"]) + "\n")
    (out_dir / f"{tag}__high.txt").write_text("\n".join(result["high_ids"]) + "\n")

    # Full assignment (low / high / mid) for every conversation this candidate could score.
    with open(out_dir / f"{tag}__assignment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["/".join(id_cols), "partition"])
        for group in ("low", "high", "mid"):
            for cid in result.get(f"{group}_ids", []):
                w.writerow([cid, group])
    return tag


def summarize(result):
    feats = "+".join(result["features"])
    s = result["score"]
    flag = "" if result.get("qualified", True) else "  [below-margin]"
    return (
        f"{feats:<55} n={result['n_used']:>4} grp={result['group_size']:>4} "
        f"d={s['cohens_d']:>6.3f} auc={s['auc']:.3f} p={s['p_value']:.1e} "
        f"z_gap={s['z_gap']:>6.3f} min_col_d={result['min_column_cohens_d']:>6.3f}{flag}"
    )


_ID_FIELDS = ("low_ids", "high_ids", "mid_ids")


def build_payload(results, params, n_candidates, include_ids):
    """Assemble the JSON-serialisable results object (run metadata + ranked splits).

    Partition member id lists (low/high/mid) are dropped unless `include_ids`, since they can
    be large. `n_qualified` counts splits that cleared the margin; `n_results` is everything
    returned (which exceeds `n_qualified` only under --all-combos).
    """
    out = []
    for r in results:
        item = {k: v for k, v in r.items() if include_ids or k not in _ID_FIELDS}
        out.append(item)
    return {
        "params": params,
        "n_candidates": n_candidates,
        "n_qualified": sum(1 for r in results if r.get("qualified", True)),
        "n_results": len(out),
        "results": out,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Conversation-level annotations CSV.")
    p.add_argument("--id-col", default=None,
                   help=f"Identifier column(s), comma-separated for a composite key. Auto-detected "
                        f"from {COMMON_ID_COLS} if omitted (extended to a unique key when the "
                        f"primary id repeats, e.g. the raw per-respondent survey).")
    p.add_argument("--label-col", default="label", help="Categorical label column (used for reporting / exclusion).")
    p.add_argument("--feature-cols", default=None,
                   help="Comma-separated columns to search over. If omitted, they are selected "
                        "dynamically from the data + data dictionary (see auto_select_features).")
    p.add_argument("--data-dict", type=Path, default=DEFAULT_DATADICT,
                   help="BetterUp data dictionary CSV; used to keep only ordinal/numeric survey "
                        "items and drop categorical / free-text columns. Pass 'none' to disable.")
    p.add_argument("--data-dict-strict", action="store_true",
                   help="Restrict auto-selection to columns the dictionary documents as "
                        "ordinal/numeric (drops derived columns absent from the dictionary).")
    p.add_argument("--exclude-cols", default="",
                   help="Comma-separated columns to exclude from auto-selection.")
    p.add_argument("--exclude-suffixes", default="",
                   help="Comma-separated column-name suffixes to drop from auto-selection, e.g. "
                        "'_std' to keep only the per-participant means in surveys_agg_by_participant.csv.")
    p.add_argument("--min-numeric-frac", type=float, default=0.8,
                   help="A column is treated as numeric if at least this fraction of its "
                        "non-empty values parse as floats.")
    p.add_argument("--list-features", action="store_true",
                   help="Print the auto-selected feature columns (and why others were skipped), then exit.")
    p.add_argument("--exclude-labels", default="UNSCORED",
                   help="Comma-separated label values to drop before searching.")
    p.add_argument("--min-raters", type=float, default=0,
                   help="Require each constituent column to have >= this many raters (needs <col>_n_raters columns).")
    p.add_argument("--group-by", default=None,
                   help="Aggregate rows sharing these column(s) into one unit before searching, "
                        "e.g. 'convo_id' to collapse the raw per-respondent survey to conversation "
                        "level. The group column(s) become the id.")
    p.add_argument("--group-agg", choices=["mean", "median", "min", "max"], default="mean",
                   help="How to combine each feature across the rows of a --group-by group.")

    p.add_argument("--max-combo", type=int, default=1, help="Search subsets up to this many columns.")
    p.add_argument("--min-combo", type=int, default=1, help="Smallest subset size to consider.")
    p.add_argument("--group-frac", type=float, default=0.25,
                   help="Fraction taken from each extreme (0.25 = bottom vs top quartile; 0.5 = median split).")
    p.add_argument("--min-group-size", type=int, default=20, help="Minimum conversations per partition.")

    p.add_argument("--margin", type=float, default=1.0, help="Required separation on --margin-metric.")
    p.add_argument("--margin-metric", choices=["cohens_d", "auc", "raw_gap"], default="cohens_d",
                   help="cohens_d: standardized mean diff; auc: 2*|AUC-0.5|; raw_gap: z-score gap between bands.")
    p.add_argument("--require-all-columns", action="store_true",
                   help="For combos, also require every constituent column to separate by >= --margin (Cohen's d).")
    p.add_argument("--alpha", type=float, default=None,
                   help="If set, also require Mann-Whitney p < alpha.")
    p.add_argument("--all-combos", action="store_true",
                   help="Keep every tried combination in the output (each tagged 'qualified'), "
                        "not only those clearing the margin -- useful for validating the search.")

    p.add_argument("--top-k", type=int, default=25, help="How many qualifying splits to print (text mode).")
    p.add_argument("--json", action="store_true",
                   help="Emit the full results as JSON to stdout instead of the text table "
                        "(diagnostics still go to stderr).")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Also write the results as JSON to this file.")
    p.add_argument("--json-ids", action="store_true",
                   help="Include the low/high partition member ids in JSON output (omitted by default).")
    p.add_argument("--write-partitions", action="store_true",
                   help="Write id lists + assignment CSV for each qualifying split into --out-dir.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "partitions",
                   help="Directory for partition files when --write-partitions is set.")
    return p.parse_args()


def resolve_features(args, rows, id_cols):
    """Return the list of feature columns to search over, explicit or auto-selected."""
    if args.feature_cols:
        return [c.strip() for c in args.feature_cols.split(",") if c.strip()], {}

    use_dict = str(args.data_dict).lower() not in ("none", "")
    dict_info = load_data_dictionary(args.data_dict) if use_dict else {}
    extra_exclude = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    exclude_suffixes = [s.strip() for s in args.exclude_suffixes.split(",") if s.strip()]
    features, skipped = auto_select_features(
        rows, id_cols, args.label_col, dict_info,
        args.data_dict_strict, extra_exclude, args.min_numeric_frac, exclude_suffixes,
    )
    src = f"data dictionary ({len(dict_info)} documented cols)" if use_dict else "data only"
    print(f"[info] auto-selected {len(features)} feature columns from {src}"
          f"{' [strict]' if args.data_dict_strict else ''}", file=sys.stderr)
    return features, skipped


def main():
    args = parse_args()
    exclude = [c.strip() for c in args.exclude_labels.split(",") if c.strip()]

    rows, id_cols = load_rows(args.metrics, args.id_col)
    features, skipped = resolve_features(args, rows, id_cols)

    if args.list_features:
        print("selected feature columns:")
        for c in features:
            print(f"    {c}")
        if skipped:
            print("\nskipped numeric columns (dictionary classification):")
            for c, why in sorted(skipped.items()):
                print(f"    {c:<40} {why}")
        return

    if not features:
        sys.exit("[error] no feature columns selected. Pass --feature-cols explicitly, relax "
                 "--data-dict-strict, or check --data-dict.")

    if args.group_by:
        group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
        missing = [c for c in group_cols if c not in rows[0]]
        if missing:
            sys.exit(f"[error] --group-by column(s) {missing} not found. Columns: {list(rows[0])}")
        n_before = len(rows)
        rows = aggregate_rows(rows, group_cols, features, args.label_col, args.group_agg)
        id_cols = group_cols
        print(f"[info] aggregated {n_before} rows -> {len(rows)} groups by {group_cols} "
              f"({args.group_agg})", file=sys.stderr)

    records = prepare(rows, id_cols, features, args.label_col, exclude, args.min_raters)
    print(f"[info] {len(records)}/{len(rows)} rows have >=1 of the {len(features)} features "
          f"(each candidate uses the rows that have all of its columns)", file=sys.stderr)
    if len(records) < 2 * args.min_group_size:
        sys.exit(f"[error] not enough rows ({len(records)}) for two groups of >= {args.min_group_size}")

    subsets = []
    hi = min(args.max_combo, len(features))
    for size in range(max(1, args.min_combo), hi + 1):
        subsets.extend(itertools.combinations(features, size))
    print(f"[info] evaluating {len(subsets)} feature subsets "
          f"(sizes {args.min_combo}..{hi}), group_frac={args.group_frac}", file=sys.stderr)

    results = []
    for subset in subsets:
        res = evaluate_subset(records, subset, args.group_frac, args.min_group_size)
        if not res:
            continue
        res["qualified"] = qualifies(res, args)
        if res["qualified"] or args.all_combos:
            results.append(res)

    results.sort(key=lambda r: margin_value(r, args.margin_metric), reverse=True)
    n_qual = sum(1 for r in results if r["qualified"])
    extra = f"; keeping all {len(results)} evaluated (--all-combos)" if args.all_combos else ""
    print(f"[info] {n_qual} of {len(subsets)} subsets qualify at "
          f"{args.margin_metric} >= {args.margin}{extra}\n", file=sys.stderr)

    params = {
        "metrics": str(args.metrics),
        "id_cols": id_cols,
        "features": features,
        "n_features": len(features),
        "group_by": id_cols if args.group_by else None,
        "group_agg": args.group_agg if args.group_by else None,
        "min_combo": args.min_combo,
        "max_combo": hi,
        "group_frac": args.group_frac,
        "min_group_size": args.min_group_size,
        "margin": args.margin,
        "margin_metric": args.margin_metric,
        "require_all_columns": args.require_all_columns,
        "alpha": args.alpha,
        "all_combos": args.all_combos,
    }

    if args.json:
        payload = build_payload(results, params, len(subsets), include_ids=args.json_ids)
        print(json.dumps(payload, indent=2))
    else:
        for res in results[: args.top_k]:
            print(summarize(res))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = build_payload(results, params, len(subsets), include_ids=args.json_ids)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[info] wrote {len(results)} qualifying splits to {args.out_json}", file=sys.stderr)

    if args.write_partitions:
        for res in results:
            tag = write_partitions(res, args.out_dir, id_cols)
        if results:
            print(f"[info] wrote partition files for {len(results)} splits to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
