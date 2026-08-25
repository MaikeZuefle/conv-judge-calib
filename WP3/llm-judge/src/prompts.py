import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "data" / "candor_questions.json"
with open(QUESTIONS_PATH) as f:
    ALL_QUESTIONS = json.load(f)


def _select(predicate):
    return [q for q in ALL_QUESTIONS if predicate(q)]


_GENERAL = _select(lambda q: not q["single_speaker"] and not q.get("extra"))
_LIKING_EXTRA = _select(lambda q: q.get("extra"))

CANDOR_QUESTIONS_SCALED = [(q["question"], tuple(q["scale"])) for q in _GENERAL]
CANDOR_QUESTIONS_LIKING_SCALED = CANDOR_QUESTIONS_SCALED + [
    (q["question"], tuple(q["scale"])) for q in _LIKING_EXTRA
]

CANDOR_QUESTIONS = [q for q, _ in CANDOR_QUESTIONS_SCALED]
CANDOR_QUESTIONS_LIKING = [q for q, _ in CANDOR_QUESTIONS_LIKING_SCALED]

# (label, scale, survey_column) for each question with human-eval metadata, in CANDOR_QUESTIONS_LIKING order
CANDOR_QUESTIONS_LIKING_META = [
    (q["label"], tuple(q["scale"]), q["survey_column"]) for q in (_GENERAL + _LIKING_EXTRA)
]


def cot_questions_prompt(questions):
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        "Listen to this conversation. Answer the following questions about it:\n\n"
        f"{questions_block}\n\n"
        "Then respond with only a JSON object with two fields: \"answers\", a list of your "
        "answers to the questions above in order, and \"score\", your rating of how successful "
        "the conversation is on a scale from 0 (not successful) to 10 (very successful)."
    )


def summary_questions_prompt(questions):
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        "Listen to this conversation. For each of the following questions, give a brief summary "
        "of the relevant parts of the conversation, rather than answering with a rating:\n\n"
        f"{questions_block}\n\n"
        "Then respond with only a JSON object with two fields: \"summaries\", a list of your "
        "question-relevant summaries in order, and \"score\", your rating of how successful the "
        "conversation is on a scale from 0 (not successful) to 10 (very successful)."
    )


def scaled_questions_prompt(questions):
    questions_block = "\n".join(f"{i}. (scale {lo}-{hi}) {q}" for i, (q, (lo, hi)) in enumerate(questions, 1))
    return (
        "Listen to this conversation. Answer the following questions about it, using the scale "
        "given in parentheses for each question:\n\n"
        f"{questions_block}\n\n"
        'Respond with only a JSON object with one field: "answers", a list of your numeric '
        "answers to the questions above, in order."
    )


CATEGORY_LABELS = {"HSC": "high success", "MSC": "medium success", "LSC": "low success"}

CATEGORY_QUESTIONS = [
    "Across the conversation as a whole, how positive or negative does the speakers' apparent "
    "mood seem overall?",
    "At the beginning of the conversation, how positive or negative does the speakers' apparent "
    "mood seem overall?",
    "Around the middle of the conversation, how positive or negative does the speakers' apparent "
    "mood seem overall?",
    "Toward the end of the conversation, how positive or negative does the speakers' apparent "
    "mood seem overall?",
    "At the most positive moment of the conversation, how positive or negative do the speakers' "
    "apparent mood seem overall?",
    "How enjoyable does the conversation appear to be for the speakers?",
    "To what extent do the two speakers appear to like each other?",
    "How well do the two speakers appear to get along with each other?",
]

# subset used for judging a single third of a conversation: the start/mid/end mood questions are
# dropped since "overall mood" already covers the whole clip when the model only hears one part
PART_QUESTIONS = [CATEGORY_QUESTIONS[i] for i in (0, 4, 5, 6, 7)]


def _questions_block(questions):
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))


def category_prompt(categories):
    options = ", ".join(f"{c} ({CATEGORY_LABELS[c]})" for c in categories)
    quoted = " or ".join(f'"{c}"' for c in categories)
    return (
        "Listen to this conversation. Answer the following questions about it:\n\n"
        f"{_questions_block(CATEGORY_QUESTIONS)}\n\n"
        "Your answers to these questions should determine how successful the conversation was "
        f"overall. Based on them, classify the conversation into exactly one of {len(categories)} "
        f"categories: {options}.\n\n"
        "Then respond with only a JSON object with two fields: \"answers\", a list of your "
        "answers to the questions above in order, and \"category\", your classification of the "
        f"conversation as one of {quoted}."
    )


def part_summary_prompt(questions):
    return (
        "Listen to this conversation. For each of the following questions, give a brief summary "
        "of the relevant parts of the conversation:\n\n"
        f"{_questions_block(questions)}\n\n"
        "Then respond with only a JSON object with one field: \"summaries\", a list of your "
        "question-relevant summaries in order."
    )


def aggregate_category_prompt(categories):
    options = ", ".join(f"{c} ({CATEGORY_LABELS[c]})" for c in categories)
    quoted = " or ".join(f'"{c}"' for c in categories)
    return (
        "Below are question-relevant summaries of three consecutive parts of a conversation, "
        "covering the speakers' mood, enjoyment, and how well they got along, given in "
        "chronological order.\n\n"
        "Based on all three parts together, respond with only a JSON object with two fields: "
        '"score", your rating of how successful the conversation is on a scale from 0 (not '
        'successful) to 10 (very successful), and "category", your classification of the '
        f"conversation into exactly one of {len(categories)} categories: {options}. Respond with "
        f'"category" as one of {quoted}.'
    )


PROMPTS = {
    "success": (
        "Listen to this conversation. Rate how successful it is on a scale "
        "from 0 (not successful) to 10 (very successful). Respond with only the number."
    ),
    "CoT_summary": (
        "Listen to this conversation. Respond with only a JSON object with two fields: "
        '"summary", a brief summary of what was discussed, and "score", your rating of how '
        "successful the conversation is on a scale from 0 (not successful) to 10 (very successful)."
    ),
    "CoT_questions": cot_questions_prompt(CANDOR_QUESTIONS),
    "CoT_questions_liking": cot_questions_prompt(CANDOR_QUESTIONS_LIKING),
    "CoT_questions_summary_liking": summary_questions_prompt(CANDOR_QUESTIONS_LIKING),
    "questions": scaled_questions_prompt(CANDOR_QUESTIONS_SCALED),
    "questions_liking": scaled_questions_prompt(CANDOR_QUESTIONS_LIKING_SCALED),
    "CoT_category": category_prompt(["HSC", "MSC", "LSC"]),
    "CoT_category_hsc_lsc": category_prompt(["HSC", "LSC"]),
}

# used by run_judge_ensemble.py's two-stage pipeline (one model summarizes, another judges the
# summary); kept out of PROMPTS since they aren't meant to be run standalone via run_judge.py
SUMMARY_ONLY_PROMPT = (
    "Listen to this conversation. Respond with only a JSON object with one field: "
    '"summary", a brief summary of what was discussed.'
)
JUDGE_FROM_SUMMARY_PROMPT = (
    "The text above is a summary of a conversation. Respond with only a JSON object with one "
    'field: "score", your rating of how successful the conversation seems to have been on a '
    "scale from 0 (not successful) to 10 (very successful)."
)
