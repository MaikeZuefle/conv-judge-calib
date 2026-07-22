import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "data" / "candor_questions.json"
with open(QUESTIONS_PATH) as f:
    CANDOR_QUESTIONS = [q["question"] for q in json.load(f) if not q["single_speaker"]]

QUESTIONS_BLOCK = "\n".join(f"{i}. {q}" for i, q in enumerate(CANDOR_QUESTIONS, 1))

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
    "CoT_questions": (
        "Listen to this conversation. Answer the following questions about it:\n\n"
        f"{QUESTIONS_BLOCK}\n\n"
        "Then respond with only a JSON object with two fields: \"answers\", a list of your "
        "answers to the questions above in order, and \"score\", your rating of how successful "
        "the conversation is on a scale from 0 (not successful) to 10 (very successful)."
    ),
}
