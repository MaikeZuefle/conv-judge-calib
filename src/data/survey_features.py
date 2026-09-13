"""Shared survey feature definitions and loading for CANDOR success scoring.

Each conversation has two survey rows (one per participant). Features are z-scored
across all responses before averaging, since the raw scales differ (1-9 Likert for
affect, 0-100 sliders for conversationalist / my_friends_like_you).
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

from paths import CANDOR_ROOT

# features used for the composite success score, equally weighted after z-scoring
FEATURES = [
    "affect",
    "overall_affect",
    "begin_affect",
    "middle_affect",
    "end_affect",
    "best_affect",
    "how_enjoyable",
    "i_like_you",
    "you_like_me",
    "conversationalist",
    "my_friends_like_you",
]

# free-text fields, printed when inspecting partner disagreement
TEXT_FIELDS = ["critical_positive", "critical_negative", "rest_of_day_open"]


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_responses(candor_root=CANDOR_ROOT):
    """Return list of per-participant response dicts, one per survey row."""
    responses = []
    for path in sorted(Path(candor_root).glob("*/survey.csv")):
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if not row.get("user_id", "").strip():
                        continue
                    responses.append(row)
        except Exception as e:
            print(f"  [WARN] {path.parent.name}: {e}")
    return responses


def zscore_stats(responses, features=FEATURES):
    """Return {feature: (mean, std)} computed across all responses."""
    stats = {}
    for feat in features:
        vals = [v for v in (_try_float(r.get(feat)) for r in responses) if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        stats[feat] = (mean, std if std > 0 else 1.0)
    return stats


def composite(response, stats, features=FEATURES, min_features=8):
    """Mean of z-scored features for one response; None if too many are missing."""
    zs = []
    for feat in features:
        val = _try_float(response.get(feat))
        if val is None or feat not in stats:
            continue
        mean, std = stats[feat]
        zs.append((val - mean) / std)
    if len(zs) < min_features:
        return None
    return sum(zs) / len(zs)


def load_transcript(convo_id, candor_root=CANDOR_ROOT):
    """Speaker-labelled transcript read from the local dataset copy, so that compute
    nodes never need hub access. Matches the formatting in data/candor.py."""
    path = Path(candor_root) / convo_id / "transcription" / "transcript_backbiter.csv"
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    speaker_ids = []
    for row in rows:
        sid = row["speaker"].strip()
        if sid not in speaker_ids:
            speaker_ids.append(sid)
    labels = {sid: f"Speaker {chr(65 + i)}" for i, sid in enumerate(speaker_ids)}

    lines = []
    for row in rows:
        line = f"{labels[row['speaker'].strip()]}: {row['utterance']}"
        if row.get("backchannel"):
            line += f" [{labels[row['backchannel_speaker'].strip()]} backchannel: {row['backchannel']}]"
        lines.append(line)
    return "\n".join(lines)


def by_conversation(responses, stats):
    """Return {convo_id: [(composite, response), ...]} for conversations with 2 scored responses."""
    convos = defaultdict(list)
    for r in responses:
        score = composite(r, stats)
        if score is not None:
            convos[r["convo_id"].strip()].append((score, r))
    return {cid: pair for cid, pair in convos.items() if len(pair) == 2}
