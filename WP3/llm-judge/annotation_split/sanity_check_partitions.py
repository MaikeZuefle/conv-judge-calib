#!/usr/bin/env python3
"""Independently recompute the separation margins of a saved partition, for sanity checking.

`annotation_split.py --write-partitions` writes, per split, a `<features>__assignment.csv` with
one row per conversation and a `low` / `high` / `mid` label. This script takes such an
assignment CSV, looks the corresponding rows up in the source data, and recomputes the
separation statistics between the `low` and `high` partitions *from scratch* -- reusing the
exact stats functions from `annotation_split.py`, so the numbers are directly comparable to
what the search reported. If they match, the saved partition is faithful.

What it computes (high vs low partition, exactly as the search does):
  * composite score = mean of z-scores over the split's features (z-scored over every assigned
    conversation, i.e. low + high + mid), then Cohen's d / AUC / Mann-Whitney p / z-gap on it;
  * per-column Cohen's d / AUC / raw mean difference for each feature (and any `--extra-cols`);
  * the label mix (e.g. HSC/MSC/LSC) of each partition.

The feature columns are read from the assignment filename (`a+b__assignment.csv` -> a, b) unless
`--feature-cols` is given. `--extra-cols` additionally reports how well the *same* partition
separates on other annotations (e.g. does an `in_common` split also separate `paper_pcs_proxy`?).

Matching the data:
  * The id column(s) are read from the assignment header (`convo_id`, or a composite like
    `convo_id/user_id`). If several source rows map to one id (e.g. the raw per-respondent survey
    split with `--group-by convo_id`), they are aggregated with `--group-agg` (default mean),
    mirroring the search. Point `--metrics` at the same source you searched.

Examples
--------
    # Recompute margins for one saved split against the default conversation-level file
    python sanity_check_partitions.py partitions/in_common__assignment.csv

    # Check every saved split in a directory, and also report separation on paper_pcs_proxy
    python sanity_check_partitions.py partitions/ --extra-cols paper_pcs_proxy

    # A split built from the raw per-respondent survey aggregated to conversation level
    python sanity_check_partitions.py partitions/how_enjoyable__assignment.csv \
        --metrics all_surveys_raw.csv --group-agg mean --json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import annotation_split as asp

_HERE = Path(__file__).resolve().parent
DEFAULT_METRICS = _HERE.parent / "src" / "data" / "survey_metrics_conversation_level.csv"
ASSIGN_SUFFIX = "__assignment.csv"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_assignment(path):
    """Return (id_cols, {id: partition}) from a <features>__assignment.csv file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            sys.exit(f"[error] {path}: expected an 'id,partition' header")
        id_cols = header[0].split("/")
        assign = {}
        for parts in reader:
            if len(parts) < 2 or not parts[0]:
                continue
            assign[parts[0]] = parts[1].strip()
    if not assign:
        sys.exit(f"[error] {path}: no assignment rows")
    return id_cols, assign


def features_from_filename(path):
    name = Path(path).name
    if name.endswith(ASSIGN_SUFFIX):
        return [c for c in name[: -len(ASSIGN_SUFFIX)].split("+") if c]
    return []


def load_values_by_id(metrics_path, id_cols, wanted_cols, group_agg, label_col):
    """Return ({id: {col: float}}, {id: label}, n_source_rows).

    Rows sharing an id (e.g. two respondents per convo_id) are reduced per column with
    `group_agg`, mirroring annotation_split's --group-by aggregation.
    """
    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[error] no rows in {metrics_path}")
    header = set(rows[0])
    missing_id = [c for c in id_cols if c not in header]
    if missing_id:
        sys.exit(f"[error] id column(s) {missing_id} not in {metrics_path}. Columns: {sorted(header)}")

    groups = {}
    for row in rows:
        rid = "/".join(row.get(c, "") for c in id_cols)
        groups.setdefault(rid, []).append(row)

    values, labels = {}, {}
    for rid, members in groups.items():
        vals = {}
        for col in wanted_cols:
            xs = [v for v in (asp._to_float(m.get(col)) for m in members) if v is not None]
            if xs:
                vals[col] = asp._reduce(xs, group_agg)
        values[rid] = vals
        if label_col:
            labs = [m.get(label_col, "") for m in members if m.get(label_col, "")]
            labels[rid] = labs[0] if labs else ""
    return values, labels, len(rows)


# --------------------------------------------------------------------------- #
# Recompute margins
# --------------------------------------------------------------------------- #
def column_stats(high_vals, low_vals):
    return {
        "high_mean": round(asp.mean(high_vals), 4) if high_vals else None,
        "low_mean": round(asp.mean(low_vals), 4) if low_vals else None,
        "raw_diff": round(asp.mean(high_vals) - asp.mean(low_vals), 4) if high_vals and low_vals else None,
        "cohens_d": round(asp.cohens_d(high_vals, low_vals), 4),
        "auc": round(asp.mann_whitney(high_vals, low_vals)[0], 4),
    }


def check_assignment(assign_path, values, labels, features, extra_cols):
    """Recompute the split's margins from the assignment membership + source values."""
    id_cols, assign = load_assignment(assign_path)

    # Rows we can actually score: assigned AND present in the source with all feature columns.
    usable = [rid for rid in assign if rid in values and all(c in values[rid] for c in features)]
    missing = [rid for rid in assign if rid not in values or not all(c in values[rid] for c in features)]

    # z-score each feature over all usable (low+high+mid) rows, then composite = mean of z.
    z = {rid: {} for rid in usable}
    for col in features:
        xs = [values[rid][col] for rid in usable]
        m, s = asp.mean(xs), asp.std(xs) or 1.0
        for rid, x in zip(usable, xs):
            z[rid][col] = (x - m) / s
    composite = {rid: asp.mean([z[rid][c] for c in features]) for rid in usable}

    groups = {g: [rid for rid in usable if assign[rid] == g] for g in ("low", "high", "mid")}
    low_c = [composite[r] for r in groups["low"]]
    high_c = [composite[r] for r in groups["high"]]

    d = asp.cohens_d(high_c, low_c)
    auc, p = asp.mann_whitney(high_c, low_c)
    z_gap = (min(high_c) - max(low_c)) if (low_c and high_c) else None

    def col_report(col):
        hi = [values[r][col] for r in groups["high"] if col in values[r]]
        lo = [values[r][col] for r in groups["low"] if col in values[r]]
        return column_stats(hi, lo)

    per_column = {col: col_report(col) for col in features}
    extra = {col: col_report(col) for col in extra_cols}

    return {
        "assignment": str(assign_path),
        "features": features,
        "id_cols": id_cols,
        "sizes": {g: len(v) for g, v in groups.items()},
        "unmatched": len(missing),
        "score": {
            "cohens_d": round(d, 4),
            "auc": round(auc, 4),
            "p_value": p,
            "z_gap": round(z_gap, 4) if z_gap is not None else None,
        },
        # A well-formed split has high-partition scores above low: d and z_gap should be > 0.
        "direction_ok": d > 0 and (z_gap is None or z_gap >= 0),
        "min_column_cohens_d": round(min((c["cohens_d"] for c in per_column.values()), default=0.0), 4),
        "per_column": per_column,
        "extra_columns": extra,
        "label_mix": {g: _label_counts(v, labels) for g, v in groups.items()},
    }


def _label_counts(ids, labels):
    counts = {}
    for rid in ids:
        lab = labels.get(rid, "")
        counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(rep):
    s = rep["score"]
    flag = "" if rep["direction_ok"] else "  !! INVERTED (high scores below low)"
    print(f"== {Path(rep['assignment']).name} ==")
    print(f"features: {'+'.join(rep['features'])}   id: {'/'.join(rep['id_cols'])}")
    print(f"sizes: low={rep['sizes']['low']} high={rep['sizes']['high']} "
          f"mid={rep['sizes']['mid']} unmatched={rep['unmatched']}")
    print(f"composite (high vs low): cohens_d={s['cohens_d']:.4f} auc={s['auc']:.4f} "
          f"p={s['p_value']:.2e} z_gap={s['z_gap']}{flag}")
    print("per-column (high vs low):")
    print(f"    {'column':<28}{'high_mean':>10}{'low_mean':>10}{'raw_diff':>10}{'cohens_d':>10}{'auc':>8}")
    for col, c in {**rep["per_column"], **rep["extra_columns"]}.items():
        tag = "  (extra)" if col in rep["extra_columns"] else ""
        print(f"    {col:<28}{_f(c['high_mean']):>10}{_f(c['low_mean']):>10}"
              f"{_f(c['raw_diff']):>10}{_f(c['cohens_d']):>10}{_f(c['auc']):>8}{tag}")
    if any(rep["label_mix"].values()):
        print(f"label_mix: low={rep['label_mix']['low']} high={rep['label_mix']['high']} "
              f"mid={rep['label_mix']['mid']}")
    print()


def _f(x):
    return f"{x:.4f}" if isinstance(x, float) else ("-" if x is None else str(x))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def collect_assignments(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.glob(f"*{ASSIGN_SUFFIX}")))
        else:
            out.append(p)
    if not out:
        sys.exit("[error] no assignment CSVs found (pass files or a directory).")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("assignments", nargs="+",
                   help="Assignment CSV file(s) or a directory containing *__assignment.csv.")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Source data CSV to read values from.")
    p.add_argument("--feature-cols", default=None,
                   help="Override the split's feature columns (default: parsed from the filename).")
    p.add_argument("--extra-cols", default="",
                   help="Comma-separated extra columns to also report separation on (e.g. paper_pcs_proxy).")
    p.add_argument("--label-col", default="label", help="Label column for the per-partition label mix.")
    p.add_argument("--group-agg", choices=["mean", "median", "min", "max"], default="mean",
                   help="How to combine multiple source rows sharing an id (mirror --group-by aggregation).")
    p.add_argument("--json", action="store_true", help="Emit the reports as JSON instead of text.")
    return p.parse_args()


def main():
    args = parse_args()
    extra_cols = [c.strip() for c in args.extra_cols.split(",") if c.strip()]
    forced_features = [c.strip() for c in args.feature_cols.split(",")] if args.feature_cols else None

    assign_paths = collect_assignments(args.assignments)

    reports = []
    for ap in assign_paths:
        features = forced_features or features_from_filename(ap)
        if not features:
            print(f"[warn] {ap}: cannot determine feature columns; use --feature-cols", file=sys.stderr)
            continue
        values, labels, _ = load_values_by_id(
            args.metrics, load_assignment(ap)[0], features + extra_cols, args.group_agg, args.label_col
        )
        reports.append(check_assignment(ap, values, labels, features, extra_cols))

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for rep in reports:
            print_report(rep)
        inverted = [Path(r["assignment"]).name for r in reports if not r["direction_ok"]]
        if inverted:
            print(f"[warn] {len(inverted)} partition(s) have inverted direction: {inverted}", file=sys.stderr)


if __name__ == "__main__":
    main()
