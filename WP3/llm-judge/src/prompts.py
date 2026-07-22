import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "data" / "candor_questions.json"
with open(QUESTIONS_PATH) as f:
    ALL_QUESTIONS = json.load(f)

CANDOR_QUESTIONS = [q["question"] for q in ALL_QUESTIONS if not q["single_speaker"] and not q.get("extra")]
CANDOR_QUESTIONS_LIKING = CANDOR_QUESTIONS + [q["question"] for q in ALL_QUESTIONS if q.get("extra")]


def questions_prompt(questions):
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        "Listen to this conversation. Answer the following questions about it:\n\n"
        f"{questions_block}\n\n"
        "Then respond with only a JSON object with two fields: \"answers\", a list of your "
        "answers to the questions above in order, and \"score\", your rating of how successful "
        "the conversation is on a scale from 0 (not successful) to 10 (very successful)."
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
    "CoT_questions": questions_prompt(CANDOR_QUESTIONS),
    "CoT_questions_liking": questions_prompt(CANDOR_QUESTIONS_LIKING),
}
