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


def cot_questions_prompt(questions):
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        "Listen to this conversation. Answer the following questions about it:\n\n"
        f"{questions_block}\n\n"
        "Then respond with only a JSON object with two fields: \"answers\", a list of your "
        "answers to the questions above in order, and \"score\", your rating of how successful "
        "the conversation is on a scale from 0 (not successful) to 10 (very successful)."
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
    "questions": scaled_questions_prompt(CANDOR_QUESTIONS_SCALED),
    "questions_liking": scaled_questions_prompt(CANDOR_QUESTIONS_LIKING_SCALED),
}
