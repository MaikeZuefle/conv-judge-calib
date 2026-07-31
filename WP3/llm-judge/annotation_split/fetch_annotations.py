#!/usr/bin/env python3
"""Fetch ONLY the CANDOR annotations from HuggingFace (no audio) and aggregate to
conversation level so `annotation_split.py` can run on them directly.

Why this is cheap
-----------------
CANDOR on the Hub (`JSALT2026-Conv-AI-Simulator/CANDOR`) is one folder per conversation:

    {convo_id}/processed/{convo_id}.mp3          <- audio, the bulk of the dataset
    {convo_id}/transcription/transcript_*.csv    <- transcripts
    {convo_id}/survey.csv                         <- the post-conversation SURVEY (annotations)

`huggingface_hub.snapshot_download(..., allow_patterns=["*/survey.csv"])` lists the repo
(a metadata-only call) and downloads *only* the survey files -- a few KB each, so the whole
annotation set is a handful of MB instead of the full multi-hundred-GB audio corpus. The
download is parallel, resumable and cached.

What it produces
----------------
Each `survey.csv` has one row per respondent (usually 2 per conversation) with the raw
per-respondent survey columns from the BetterUp data dictionary. This script aggregates the
respondents to a single row per conversation (default: mean of every numeric column) and
writes a wide conversation-level table keyed by `convo_id`, e.g.:

    convo_id,n_respondents,how_enjoyable,i_like_you,in_common,conversationalist,...

Then run the partitioner on it:

    python annotation_split.py \
        --metrics candor_annotations_conversation_level.csv \
        --feature-cols how_enjoyable,i_like_you,in_common,conversationalist \
        --max-combo 3 --margin 1.0

Notes / caveats
---------------
* The curated `label` (HSC/LSC/MSC) and `paper_pcs_proxy` in the in-repo file
  `../src/data/survey_metrics_conversation_level.csv` are DERIVED and are NOT present in the
  raw per-conversation surveys. If you need those for the full dataset, look for a
  repo-level metadata file first (`--list-only` prints every non-audio file so you can spot
  one). This script deliberately does not try to re-derive `paper_pcs_proxy`.
* Averaging the two respondents turns directional items (e.g. `i_like_you` = "how much I
  liked my partner") into a symmetric conversation-level quantity (mutual liking). Use
  `--agg first` or `--agg both` if you'd rather not average.

Requirements: `huggingface_hub` (present in the `jsalt_env` conda env). If the dataset is
gated, authenticate first: `huggingface-cli login` or set `HF_TOKEN`.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ID = "JSALT2026-Conv-AI-Simulator/CANDOR"


def _hub():
    try:
        import huggingface_hub  # noqa: F401
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("[error] huggingface_hub not installed. Activate jsalt_env or `pip install huggingface_hub`.")
    return HfApi, snapshot_download


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def list_annotation_files(repo_id):
    HfApi, _ = _hub()
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    # Everything that is not audio -- so a repo-level metadata file would show up here too.
    non_audio = [f for f in files if not f.lower().endswith((".mp3", ".wav", ".flac", ".m4a"))]
    surveys = [f for f in files if f.endswith("/survey.csv") or f == "survey.csv"]
    return files, non_audio, surveys


def cmd_list(args):
    all_files, non_audio, surveys = list_annotation_files(args.repo_id)
    print(f"[info] repo {args.repo_id}: {len(all_files)} files total")
    print(f"[info] {len(surveys)} per-conversation survey.csv files")
    print(f"[info] {len(non_audio)} non-audio files. Root/metadata candidates (not under a convo folder):")
    for f in sorted(non_audio):
        if "/" not in f or f.count("/") == 1 and not f.endswith("survey.csv"):
            print(f"    {f}")
    # Show a couple of survey paths as a sanity check.
    for f in surveys[:3]:
        print(f"[example] {f}")


# --------------------------------------------------------------------------- #
# Download (annotations only)
# --------------------------------------------------------------------------- #
def download_surveys(repo_id, local_dir, pattern, max_workers):
    _, snapshot_download = _hub()
    print(f"[info] downloading pattern {pattern!r} from {repo_id} -> {local_dir}", file=sys.stderr)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(local_dir),
        max_workers=max_workers,
    )
    return sorted(Path(local_dir).glob("**/survey.csv"))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _to_float(v):
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def aggregate_survey(path, convo_id, agg):
    """Collapse one conversation's survey.csv (>=1 respondent rows) to a flat dict.

    Numeric columns are aggregated per `agg`; the convo id and respondent count are added.
    Returns a list of output rows (one row normally; two rows when agg == 'both').
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    columns = list(rows[0].keys())
    numeric_cols = [c for c in columns if c not in ("convo_id", "user_id", "partner_id")]

    if agg == "both":
        out = []
        for i, r in enumerate(rows):
            item = {"convo_id": convo_id, "respondent": i, "n_respondents": len(rows)}
            for c in numeric_cols:
                v = _to_float(r.get(c))
                if v is not None:
                    item[c] = v
            out.append(item)
        return out

    agg_row = {"convo_id": convo_id, "n_respondents": len(rows)}
    for c in numeric_cols:
        vals = [_to_float(r.get(c)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if agg == "mean":
            agg_row[c] = sum(vals) / len(vals)
        elif agg == "first":
            agg_row[c] = vals[0]
        elif agg == "min":
            agg_row[c] = min(vals)
        elif agg == "max":
            agg_row[c] = max(vals)
        else:
            raise ValueError(agg)
    return [agg_row]


def convo_id_from(path, row0):
    # Prefer the survey's own convo_id column; fall back to the parent folder name.
    cid = (row0 or {}).get("convo_id", "").strip()
    return cid or path.parent.name


def build_table(survey_paths, agg, limit):
    if limit:
        survey_paths = survey_paths[:limit]
    all_rows = []
    numeric_seen = defaultdict(int)
    for path in survey_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        cid = convo_id_from(path, first)
        for item in aggregate_survey(path, cid, agg):
            all_rows.append(item)
            for k, v in item.items():
                if isinstance(v, float):
                    numeric_seen[k] += 1
    return all_rows, numeric_seen


def write_table(rows, numeric_seen, out_path):
    if not rows:
        sys.exit("[error] no survey rows aggregated -- nothing to write.")
    # Column order: ids first, then numeric columns sorted by how many conversations have them.
    lead = [c for c in ("convo_id", "respondent", "n_respondents") if any(c in r for r in rows)]
    numeric = [c for c in sorted(numeric_seen, key=lambda k: (-numeric_seen[k], k))
               if c not in lead]
    header = lead + numeric
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[info] wrote {len(rows)} rows x {len(header)} cols -> {out_path}", file=sys.stderr)
    # Report the most complete numeric columns as suggested --feature-cols.
    top = [c for c in numeric if numeric_seen[c] >= 0.9 * len(rows)][:15]
    if top:
        print(f"[hint] well-populated numeric columns for --feature-cols:\n    {','.join(top)}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--list-only", action="store_true",
                   help="Only list annotation/non-audio files in the repo, then exit.")
    p.add_argument("--pattern", default="*/survey.csv",
                   help="allow_patterns glob for the download (default fetches only surveys).")
    p.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parent / "candor_surveys_raw",
                   help="Where the downloaded survey.csv tree is stored.")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "candor_annotations_conversation_level.csv",
                   help="Aggregated conversation-level output CSV.")
    p.add_argument("--agg", choices=["mean", "first", "min", "max", "both"], default="mean",
                   help="How to combine the (usually 2) respondents per conversation.")
    p.add_argument("--max-workers", type=int, default=8, help="Parallel download workers.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N conversations (0 = all).")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip downloading; aggregate whatever survey.csv files are already under --raw-dir.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_only:
        cmd_list(args)
        return

    if args.skip_download:
        survey_paths = sorted(args.raw_dir.glob("**/survey.csv"))
        if not survey_paths:
            sys.exit(f"[error] no survey.csv found under {args.raw_dir}; drop --skip-download to fetch.")
    else:
        survey_paths = download_surveys(args.repo_id, args.raw_dir, args.pattern, args.max_workers)
    print(f"[info] {len(survey_paths)} survey.csv files available under {args.raw_dir}", file=sys.stderr)

    rows, numeric_seen = build_table(survey_paths, args.agg, args.limit)
    write_table(rows, numeric_seen, args.out)
    print(f"\nNext:\n    python annotation_split.py --metrics {args.out} "
          f"--feature-cols how_enjoyable,i_like_you,in_common,conversationalist --max-combo 3 --margin 1.0")


if __name__ == "__main__":
    main()
