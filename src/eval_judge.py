import argparse
import json
import re

import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analysis.plots import plot_distributions
from prompts import CANDOR_QUESTIONS_LIKING_SCALED, CANDOR_QUESTIONS_SCALED
from utils import get_examples_path, get_run_dir, log, parse_json_dict

QUESTION_SCALES = {
    "questions": [scale for _, scale in CANDOR_QUESTIONS_SCALED],
    "questions_liking": [scale for _, scale in CANDOR_QUESTIONS_LIKING_SCALED],
}

# prompts that output a discrete category instead of a continuous score, mapped to the
# categories each prompt actually offers the model: AUC is computed from the category prediction
CATEGORY_PROMPTS = {
    "CoT_category": ["HSC", "MSC", "LSC"],
    "CoT_category_hsc_lsc": ["HSC", "LSC"],
    "segmented_category_hsc_lsc": ["HSC", "LSC"],
}
CATEGORY_TO_SCORE = {"LSC": 0.0, "MSC": 5.0, "HSC": 10.0}

# of the category prompts, the ones that also ask for a separate numeric "score": Spearman and
# Krippendorff are computed from that score instead of being skipped
CATEGORY_PROMPTS_WITH_SCORE = {"segmented_category_hsc_lsc"}


def parse_single_score(output):
    data = parse_json_dict(output)
    if data is not None and "score" in data:
        try:
            return float(data["score"])
        except (TypeError, ValueError):
            pass

    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", output))
    return float(matches[-1].group()) if matches else None


def parse_questions_score(output, scales):
    data = parse_json_dict(output)
    answers = data.get("answers") if data is not None else None
    if not isinstance(answers, list) or len(answers) != len(scales):
        return None

    normalized = []
    for answer, (lo, hi) in zip(answers, scales):
        try:
            normalized.append((float(answer) - lo) / (hi - lo))
        except (TypeError, ValueError):
            return None

    return sum(normalized) / len(normalized) * 10


def parse_category_score(output, allowed_categories):
    data = parse_json_dict(output)
    category = data.get("category") if data is not None else None
    if category not in allowed_categories:
        return None
    return CATEGORY_TO_SCORE[category]


def parse_score(output, prompt):
    if prompt in CATEGORY_PROMPTS:
        return parse_category_score(output, CATEGORY_PROMPTS[prompt])
    scales = QUESTION_SCALES.get(prompt)
    if scales is not None:
        return parse_questions_score(output, scales)
    return parse_single_score(output)


def main(args):
    examples_path = get_examples_path(args)
    with open(examples_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.categories is not None:
        keep = set(args.categories.split(","))
        rows = [row for row in rows if row["category"] in keep]
        log("INFO", f"filtered to categories {sorted(keep)}: {len(rows)} examples remain")

    has_score_field = args.prompt in CATEGORY_PROMPTS_WITH_SCORE

    scores, labels, categories = [], [], []
    numeric_scores, numeric_labels = [], []
    scored_rows = []
    n_unparsed = 0
    for row in rows:
        score = parse_score(row["output"], args.prompt)
        scored_rows.append({**row, "extracted_label": score})
        if score is None:
            n_unparsed += 1
            continue
        scores.append(score)
        labels.append(row["label"])
        categories.append(row["category"])

        if has_score_field:
            numeric_score = parse_single_score(row["output"])
            if numeric_score is not None:
                numeric_scores.append(numeric_score)
                numeric_labels.append(row["label"])

    if n_unparsed:
        log("WARN", f"could not parse a score from {n_unparsed}/{len(rows)} outputs")
    if has_score_field and len(numeric_scores) < len(scores):
        log("WARN", f"could not parse a numeric score from {len(scores) - len(numeric_scores)}/{len(scores)} outputs")

    if has_score_field:
        correlation_scores, correlation_labels = numeric_scores, numeric_labels
    elif args.prompt in CATEGORY_PROMPTS:
        correlation_scores, correlation_labels = [], []
    else:
        correlation_scores, correlation_labels = scores, labels

    rho, p, alpha = None, None, None
    if correlation_scores:
        rho, p = spearmanr(correlation_scores, correlation_labels)
        log("INFO", f"spearman correlation vs paper_pcs_proxy (n={len(correlation_scores)}): {rho:.3f} (p={p:.3g})")

        # krippendorff needs a small discrete value domain
        rounded_scores = [round(s, 1) for s in correlation_scores]
        rounded_labels = [round(l * 10, 1) for l in correlation_labels]
        alpha = krippendorff.alpha([rounded_scores, rounded_labels], level_of_measurement="ordinal")
        log("INFO", f"krippendorff's alpha vs paper_pcs_proxy (n={len(correlation_scores)}): {alpha:.3f}")

    auc_by_category = {}
    for category in sorted(set(categories)):
        binary = [1 if c == category else 0 for c in categories]
        n_positive = sum(binary)
        if n_positive == 0 or n_positive == len(binary):
            continue
        auc = roc_auc_score(binary, scores)
        auc_by_category[category] = {"auc": float(auc), "n": n_positive}
        log("INFO", f"AUC ({category} vs rest, n={n_positive}/{len(binary)}): {auc:.3f}")

    run_dir = get_run_dir(args)
    with open(run_dir / "scored_examples.jsonl", "w") as f:
        for row in scored_rows:
            f.write(json.dumps(row) + "\n")

    results = {
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p": float(p) if p is not None else None,
        "krippendorff_alpha": float(alpha) if alpha is not None else None,
        "n": len(scores),
        "n_unparsed": n_unparsed,
        "auc_by_category": auc_by_category,
    }
    with open(run_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_distributions(scores, labels, run_dir / "distribution.png")

    log("INFO", f"results written to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="candor")
    parser.add_argument("--model", default="qwen25omni")
    parser.add_argument("--prompt", default="success")
    parser.add_argument("--input_modality", default="speech", choices=["speech", "text"])
    parser.add_argument("--output_folder", default="outputs")
    parser.add_argument("--categories", default=None, help="comma-separated category allowlist, e.g. HSC,LSC")
    args = parser.parse_args()
    main(args)
