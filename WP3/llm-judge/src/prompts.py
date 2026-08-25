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


_BWS_TASK = (
    "Above are {n} conversation transcripts between two speakers. Read all of them "
    "carefully. The BEST conversation is the most successful, most engaging and most "
    "positive; the WORST is the least successful, least engaging and most negative."
)

# Output formats are described rather than shown. A worked example containing a concrete
# permutation gets copied verbatim instead of answered, which silently reduces the task
# to reciting the example.
BWS_STYLES = {
    "rank": (
        'Respond with only a JSON object with one field, "ranking": the conversation '
        "numbers 1 to {n}, each appearing exactly once, ordered so that the best "
        "conversation comes first and the worst comes last."
    ),
    "cot": (
        "First, for each conversation in turn, write one sentence on how well it went. "
        "Then give your answer as a JSON object with one field, \"ranking\": the "
        "conversation numbers 1 to {n}, each appearing exactly once, ordered so that the "
        "best conversation comes first and the worst comes last."
    ),
    "best_only": (
        "Respond with only the number of the single best conversation, and nothing else."
    ),
}


def bws_prompt(n, style="rank"):
    """Instruction half of a best-worst-scaling prompt; the transcripts are passed as
    content. `style` selects the output format -- see BWS_STYLES."""
    return f"{_BWS_TASK.format(n=n)}\n\n{BWS_STYLES[style].format(n=n)}"


# A neutral system prompt. The model default ("Only return the answer requested. Do not
# include any explanation") suppresses reasoning, so it contradicts chain-of-thought.
NEUTRAL_SYSTEM = "You are a careful evaluator of conversations between people."

_OPENING = (
    "You will be shown {n} transcripts of separate conversations. In each one, two people "
    "who had never met before were asked to talk with each other for a few minutes.\n\n"
    "Read all {n} transcripts carefully. Afterwards you will be asked to judge how much "
    "the participants themselves enjoyed each conversation."
)

_CLOSING_ENJOY = (
    "Which of the {n} conversations above did its own participants most likely enjoy the "
    "most -- the one where they were most engaged with each other, warmest, and got on "
    "best?\n\nAnswer with only the number of that conversation, and nothing else."
)

# Reasoning styles must end with an explicit marker. Without one the answer cannot be
# told apart from the numerals that occur throughout the reasoning itself -- list
# markers like "1." and phrases like "Conversation 1" both parse as an answer.
_ANSWER_MARKER = (
    "\n\nWhen you have finished reasoning, end your reply with a final line of exactly "
    "the form:\nANSWER: <number>"
)

# (preamble, closing, system). A preamble states the task before the transcripts, so the
# model is not reading many thousands of tokens without knowing what it is looking for.
BWS_LAYOUTS = {
    # instruction only after the transcripts -- the original arrangement
    "closing_rank": (None, lambda n: bws_prompt(n, "rank"), None),
    "closing_best": (None, lambda n: bws_prompt(n, "best_only"), None),
    # task stated up front, output format restated at the end
    "opening_best": (lambda n: _OPENING.format(n=n),
                     lambda n: BWS_STYLES["best_only"].format(n=n), None),
    "opening_best_neutral": (lambda n: _OPENING.format(n=n),
                             lambda n: BWS_STYLES["best_only"].format(n=n), NEUTRAL_SYSTEM),
    # framed around what the participants felt, which is what the survey actually measured
    "enjoy": (lambda n: _OPENING.format(n=n),
              lambda n: _CLOSING_ENJOY.format(n=n), NEUTRAL_SYSTEM),
    "enjoy_cot": (lambda n: _OPENING.format(n=n),
                  lambda n: ("Briefly weigh up how much the participants seemed to enjoy each "
                             "conversation. " + _CLOSING_ENJOY.format(n=n) + _ANSWER_MARKER),
                  NEUTRAL_SYSTEM),
    # for native thinking mode: same question, answer marker, no explicit CoT instruction
    "enjoy_marked": (lambda n: _OPENING.format(n=n),
                     lambda n: _CLOSING_ENJOY.format(n=n) + _ANSWER_MARKER,
                     NEUTRAL_SYSTEM),
}


def bws_layout(n, layout):
    """Return (preamble, closing, system) for a named layout; None where not used."""
    preamble, closing, system = BWS_LAYOUTS[layout]
    return (preamble(n) if preamble else None, closing(n), system)


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

_FEATURE_CLOSING = (
    "After their conversation, each participant was asked this question about it:\n\n"
    "    \"{question}\"\n\n"
    "In which of the {n} conversations above would the participants most likely have given "
    "a HIGHER answer to that question?\n\nAnswer with only the number of that "
    "conversation, and nothing else."
)


def feature_layout(n, question, cot=False):
    """(preamble, closing, system) for judging one specific survey feature.

    The judge is asked the same question the participants were asked, so the target is
    the annotation itself rather than a general notion of a good conversation."""
    closing = _FEATURE_CLOSING.format(question=question, n=n)
    if cot:
        closing = ("Briefly weigh up how each conversation's participants would have answered. "
                   + closing + _ANSWER_MARKER)
    return _OPENING.format(n=n), closing, NEUTRAL_SYSTEM


_NATURALNESS_OPENING = (
    "You will be shown {n} recorded conversations between a person and an AI voice agent."
)

_NATURALNESS_CLOSING = (
    "Which of the {n} conversations felt MORE NATURAL -- more like listening to two people "
    "talking, rather than to a machine?\n\nAnswer with only the number of that "
    "conversation, and nothing else."
)


def naturalness_layout(n):
    """(preamble, closing, system) for judging which human-AI conversation felt natural."""
    return (_NATURALNESS_OPENING.format(n=n),
            _NATURALNESS_CLOSING.format(n=n),
            NEUTRAL_SYSTEM)


VOICEARENA_SYSTEM = "You are a careful evaluator of customer service phone calls."

_VOICEARENA_OPENING = (
    "You will be shown transcripts of {n} recorded phone calls to an airline's customer "
    "service line. In each call, a caller works with an agent to complete a booking task "
    "-- booking a flight, or rescheduling an existing booking.\n\n"
    "Each line is labelled with who spoke: the Caller, or the Agent.\n\n"
    "Read all {n} transcripts carefully. Afterwards you will be asked to compare the "
    "AGENTS -- not the callers."
)

# Kept deliberately close to the wording the human raters voted on, so the judge is
# scored against the question the humans actually answered.
_VOICEARENA_CLOSING = {
    "task_capability": (
        "In which of the {n} calls did the AGENT handle the caller's task BETTER -- "
        "understanding what was asked, getting the details right, and actually getting "
        "the booking done?\n\nAnswer with only the number of that call, and nothing else."
    ),
    "humanness": (
        "In which of the {n} calls did the AGENT come across as MORE HUMAN -- more like a "
        "real person on the phone, rather than an automated system?\n\nAnswer with only "
        "the number of that call, and nothing else."
    ),
}

VOICEARENA_DIMENSIONS = tuple(_VOICEARENA_CLOSING)


_VOICEARENA_OPENING_AUDIO = (
    "You will hear {n} recorded phone calls to an airline's customer service line. "
    "In each call, a caller works with an agent to complete a booking task -- "
    "booking a flight, or rescheduling an existing booking.\n\n"
    "Listen carefully to all {n} calls. Afterwards you will be asked to compare the "
    "AGENTS -- not the callers."
)


_VOICEARENA_COT_PREFIX = {
    "task_capability": "Briefly consider how well each AGENT understood the caller's request, got the details right, and completed the task. ",
    "humanness": "Briefly consider how natural, spontaneous, and human-like each AGENT sounded during the call. ",
}

def voicearena_layout(n, dimension, modality="text", cot=False):
    """(preamble, closing, system) for comparing the agents in two VoiceArena calls.

    `dimension` selects which of the two questions the raters answered is asked here;
    they are scored separately because a judge can plausibly track one and not the other.
    `modality` switches the preamble between transcript ("text") and audio ("speech") framing.
    `cot` prepends a brief reasoning instruction and appends an ANSWER marker so the
    parser can extract the final answer from generated reasoning text.
    """
    if dimension not in _VOICEARENA_CLOSING:
        raise ValueError(f"unknown dimension {dimension!r}; "
                         f"expected one of {sorted(_VOICEARENA_CLOSING)}")
    opening = _VOICEARENA_OPENING_AUDIO if modality == "speech" else _VOICEARENA_OPENING
    closing = _VOICEARENA_CLOSING[dimension].format(n=n)
    if cot:
        closing = _VOICEARENA_COT_PREFIX[dimension] + closing + _ANSWER_MARKER
    return (opening.format(n=n), closing, VOICEARENA_SYSTEM)


# ── pointwise (absolute) scoring ─────────────────────────────────────────────
# One call gets one score, instead of two calls being compared. Validation then checks
# whether score(winner) > score(loser) on the pairs humans voted on. This is far cheaper
# than pairwise (one audio per prompt rather than two) and sidesteps presentation-order
# bias entirely, at the cost of no longer asking the judge the same question the humans
# were asked -- which is exactly what the pairwise validation is there to check.

_VOICEARENA_SCORE_OPENING_AUDIO = (
    "You will hear one recorded phone call to an airline's customer service line. "
    "A caller works with an agent to complete a booking task.\n\n"
    "Listen carefully to the whole call. Afterwards you will be asked to rate the "
    "AGENT -- not the caller."
)

_VOICEARENA_SCORE_OPENING_TEXT = (
    "You will be shown a transcript of one recorded phone call to an airline's customer "
    "service line. A caller works with an agent to complete a booking task.\n\n"
    "Each line is labelled with who spoke: the Caller, or the Agent.\n\n"
    "Read the transcript carefully. Afterwards you will be asked to rate the "
    "AGENT -- not the caller."
)

_VOICEARENA_SCENARIO = (
    "\n\nThe task the caller is trying to complete is: {scenario}."
)

# Anchored at both ends so the scale does not collapse into the 7-8 band that unanchored
# 1-10 prompts tend to produce. Wording tracks the pairwise questions above, so the two
# judging modes stay comparable.
_VOICEARENA_SCORE_CLOSING = {
    "task_capability": (
        "How well did the AGENT handle the caller's task -- understanding what was "
        "asked, getting the details right, and actually getting the booking done?\n\n"
        "Rate on a scale from 1 to 10, where 1 means the agent failed the task badly "
        "(misunderstood what was asked, got details wrong, or never completed the "
        "booking) and 10 means the agent handled it flawlessly.\n\n"
        "Answer with only a single number from 1 to 10, and nothing else."
    ),
    "humanness": (
        "How human did the AGENT come across -- how much like a real person on the "
        "phone, rather than an automated system?\n\n"
        "Rate on a scale from 1 to 10, where 1 means the agent sounded entirely robotic "
        "and machine-like and 10 means the agent was indistinguishable from a real "
        "person.\n\n"
        "Answer with only a single number from 1 to 10, and nothing else."
    ),
}

# Public counterpart to VOICEARENA_DIMENSIONS, for the pointwise score layout.
VOICEARENA_SCORE_DIMENSIONS = tuple(_VOICEARENA_SCORE_CLOSING)


# Domain-general conversation judge: observations first, holistic rating second.
#
# Nothing here names a domain. The dataset supplies who is being rated (`target`) and,
# optionally, one line of context; the questions themselves are about properties any
# spoken conversation has, so the same judge runs on chitchat and on task-oriented
# dialogue without editing the prompt.
#
# The sub-questions ask only for observations, never for the verdict. Asking "how obvious
# is it that this is automated" would both leak the answer on the validation set and
# stop working between two synthetic speakers, where that question has no gradation.
# Whether the observations predict human preference is the empirical question; supplying
# the answer key would make it unanswerable.
#
# Wording is also pointed at events rather than impressions ("how often did X happen"
# rather than "how well did X go"). The first attempt asked "did the speaker do X well?"
# five times and got 9/9 on every question for every system -- no signal reached the
# final score.
JUDGE_DIMENSIONS = ("task_capability", "humanness")

_COT_QUESTIONS = {
    "task_capability": [
        ("Repetition",
         "How often did the other speaker have to repeat or re-state information {target} "
         "already had? (1 = repeatedly, 10 = never)"),
        ("Read-back accuracy",
         "When {target} restated or confirmed information back, how often was it wrong? "
         "(1 = wrong most times, 10 = always correct)"),
        ("Kept track",
         "When the other speaker changed or corrected something, did {target} keep track "
         "of what the current state was? (1 = lost track, 10 = tracked it exactly)"),
        ("Accomplished goal",
         "Did {target} accomplish what the other speaker was trying to achieve? "
         "(1 = not at all, 10 = fully)"),
        ("Wasted exchanges",
         "How much of the conversation was spent on exchanges that did not move things "
         "forward? (1 = most of it, 10 = almost none)"),
    ],
    "humanness": [
        ("Prosodic variation",
         "How much did {target}'s pitch, pace and emphasis vary across the conversation? "
         "(1 = flat and uniform, 10 = highly varied)"),
        ("Disfluency",
         "How often did {target} hesitate, restart a sentence, or use filler words? "
         "(1 = never, 10 = frequently)"),
        ("Response timing",
         "How natural was {target}'s response timing? (1 = consistently too fast, or gaps "
         "of unnatural length, 10 = consistently natural)"),
        ("Adaptivity",
         "When the other speaker interrupted or changed direction, how much did {target} "
         "adapt what they were saying? (1 = carried on unchanged, 10 = fully adapted)"),
        ("Phrasing variety",
         "How varied was {target}'s phrasing across similar moments in the conversation? "
         "(1 = repeated the same formulas, 10 = phrased things freshly each time)"),
    ],
}

# Variant: efficiency-focused task_capability.
# The original CoT audio correlates heavily with call duration (Spearman +0.395) because
# the model interprets a longer call as a more thorough agent. These questions directly
# ask about brevity and error rate, pushing against that correlation.
_COT_QUESTIONS_EFFICIENT = {
    "task_capability": [
        ("Repetition",
         "How many times did the other speaker have to repeat the same information to "
         "{target}? (1 = many times, 10 = never)"),
        ("Read-back accuracy",
         "When {target} read back or confirmed details, how accurate was it? "
         "(1 = mostly wrong, 10 = always correct)"),
        ("Error recovery",
         "When the other speaker corrected {target}, did it apply the correction cleanly "
         "the first time? (1 = ignored or re-made the same error, 10 = applied immediately)"),
        ("Efficiency",
         "Did {target} complete the task in a reasonable number of exchanges -- without "
         "unnecessary filler, re-asks, or loops? (1 = far too many exchanges, 10 = efficient)"),
        ("Accomplished goal",
         "Was the caller's request fully resolved by the end of the call? "
         "(1 = not at all, 10 = completely)"),
    ],
}

# Variant: acoustic-focused humanness (audio only).
# The standard humanness questions get a human-AI gap of +0.038 -- essentially zero.
# These questions target concrete physical-acoustic events that TTS systems handle
# differently from real humans: breath sounds, mid-utterance quality shifts, dysfluencies
# that aren't just fillers, rhythm breaks, and emotional tone drift.
_COT_QUESTIONS_ACOUSTIC = {
    "humanness": [
        ("Breath and mouth sounds",
         "Did you hear any audible breath sounds, lip smacks, or mouth noise from {target}? "
         "(1 = none whatsoever, 10 = frequently present)"),
        ("Voice quality shifts",
         "Did {target}'s voice quality or resonance shift within a single turn -- "
         "going slightly breathy, pressed, or rough? (1 = no shift, perfectly consistent, "
         "10 = noticeable shifts)"),
        ("Dysfluency events",
         "Did {target} ever restart mid-word, repeat a syllable, or fill a gap with 'uh' "
         "or 'um' in a way that sounded unrehearsed? (1 = never, 10 = several times)"),
        ("Rhythm breaks",
         "Were there any unexpected pauses or hesitations inside {target}'s turns that "
         "broke the rhythm -- not silence between turns, but inside an utterance? "
         "(1 = none, 10 = several)"),
        ("Emotional drift",
         "Did {target}'s warmth, energy, or emotional coloring noticeably shift across "
         "the call? (1 = completely flat and even throughout, 10 = varied naturally)"),
    ],
}

# Variant: scripted-vs-spontaneous humanness (text only).
# Text humanness inverts: AI agents score higher than humans because transcripts strip
# disfluencies and AI agents produce cleaner language. These questions focus on what
# remains detectable in text: formulaic scripted phrases vs. spontaneous language.
_COT_QUESTIONS_SCRIPTED = {
    "humanness": [
        ("Scripted phrases",
         "How many of {target}'s responses sounded like fixed templates or scripts -- "
         "phrases like 'I'd be happy to help with that' or 'Thank you for your patience'? "
         "(1 = almost every turn, 10 = very few or none)"),
        ("Unexpected language",
         "Did {target} ever say something that surprised you -- an unusual word choice, "
         "a detour from the expected path, or a personal remark? "
         "(1 = no, everything was predictable, 10 = yes, several moments)"),
        ("Self-correction",
         "Did {target} ever correct themselves, restart a sentence, or revise a statement "
         "they had just made? (1 = never, 10 = several times)"),
        ("Response variety",
         "When {target} said 'yes', 'understood', or 'okay' across the call, did the "
         "wording vary, or was it always the same phrase? "
         "(1 = identical phrase every time, 10 = varied each time)"),
        ("Off-script moment",
         "Did {target} ever acknowledge something specific the caller said rather than "
         "returning immediately to the task? (1 = never, 10 = yes, several times)"),
    ],
}

_COT_CLOSING = {
    "task_capability": (
        "Then, taking your answers into account, rate how well {target} handled what the "
        "other speaker needed, overall."
    ),
    "humanness": (
        "Then, taking your answers into account, rate how much {target} sounded like a "
        "real person rather than an automated system, overall."
    ),
}

JUDGE_SYSTEM = "You are a careful evaluator of recorded conversations."

_OPENING_AUDIO = (
    "You will hear a recorded conversation between two speakers. Listen to the whole "
    "recording. Afterwards you will be asked to rate {target} -- not the other speaker."
)
_OPENING_TEXT = (
    "You will be shown a transcript of a conversation between two speakers. Each line is "
    "labelled with who spoke. Read the whole transcript. Afterwards you will be asked to "
    "rate {target} -- not the other speaker."
)
_CONTEXT_LINE = "\n\nContext: {context}."


def _opening(target, modality, context):
    text = (_OPENING_AUDIO if modality == "speech" else _OPENING_TEXT).format(target=target)
    if context:
        text += _CONTEXT_LINE.format(context=context)
    return text


_COT_QUESTION_VARIANTS = {
    ("task_capability", "efficient"): _COT_QUESTIONS_EFFICIENT["task_capability"],
    ("humanness", "acoustic"):        _COT_QUESTIONS_ACOUSTIC["humanness"],
    ("humanness", "scripted"):        _COT_QUESTIONS_SCRIPTED["humanness"],
}


def cot_layout(dimension, target="the agent", modality="speech", context=None,
               variant=None):
    """(preamble, closing, system) for the observation questions.

    `target` names the speaker under judgement in the dataset's own terms ("the agent",
    "Speaker B"); `context` is one optional line, e.g. the goal in a task-oriented set.
    `variant` selects an alternative sub-question set: "efficient" (task_capability),
    "acoustic" or "scripted" (humanness). None uses the default questions.
    The holistic rating is elicited separately -- see cot_score_closing.
    """
    if dimension not in _COT_QUESTIONS:
        raise ValueError(f"unknown dimension {dimension!r}; "
                         f"expected one of {sorted(_COT_QUESTIONS)}")
    questions = _COT_QUESTION_VARIANTS.get((dimension, variant),
                                           _COT_QUESTIONS[dimension])
    block = "\n".join(f"{i}. {q.format(target=target)}"
                      for i, (_, q) in enumerate(questions, 1))
    closing = (
        f"Answer the following questions about {target}, each on a scale from 1 to 10:\n\n"
        f"{block}\n\n"
        "Write one short line per question, in order, as '<number>. <rating> - "
        "<a few words of justification>'."
    )
    return _opening(target, modality, context), closing, JUDGE_SYSTEM


def cot_score_closing(dimension, target="the agent"):
    """The holistic rating, asked once the observations have been generated."""
    return (
        _COT_CLOSING[dimension].format(target=target)
        + "\n\nRate on a scale from 1 to 10. Answer with only a single number from 1 to 10, "
          "and nothing else."
    )


def binary_human_layout(target="the agent", modality="speech", context=None):
    """(preamble, closing, system) for the single-question ablation.

    One question, one word out. If this beats the graded judge, the rating scale is the
    problem rather than the model's ability to tell the two apart.
    """
    closing = (
        f"Is {target} a real human being, rather than an automated system?\n\n"
        "Answer with only one word: Yes or No."
    )
    return _opening(target, modality, context), closing, JUDGE_SYSTEM


COT_DIMENSION_QUESTIONS = {d: [label for label, _ in qs] for d, qs in _COT_QUESTIONS.items()}


def humanai_naturalness_layout(target="the agent", modality="speech", context=None):
    """(preamble, closing, system) for rating one human-AI conversation 1-10.

    The closing deliberately echoes the wording the study put to its own participants
    ("how natural did this conversation feel, did it feel like talking to a person or
    not?") so the judge is answering the same question the human rating answers. The
    participants used a 1-5 scale; 1-10 is used here because the score is read from the
    model's distribution over the score tokens, and a wider scale separates
    conversations that a 5-point scale ties together. Correlation is rank-based, so the
    difference in range does not matter.
    """
    closing = (
        f"Overall, how natural did this conversation feel -- did it feel like {target} "
        "was a person rather than an automated system?\n\n"
        "Rate on a scale from 1 to 10, where 1 is completely artificial and 10 is "
        "indistinguishable from a person. Answer with only a single number from 1 to 10, "
        "and nothing else."
    )
    return _opening(target, modality, context), closing, JUDGE_SYSTEM


def voicearena_score_layout(dimension, modality="speech", scenario=None):
    """(preamble, closing, system) for rating the agent in ONE VoiceArena call 1-10.

    `scenario` is the call's task description from metadata (e.g. "Book a Flight for Two
    People"). Task capability is not well defined without knowing the task, so it is
    included when available; the outcome from the tool log is deliberately NOT given,
    so the judge has to assess the interaction rather than read off whether it worked.
    """
    if dimension not in _VOICEARENA_SCORE_CLOSING:
        raise ValueError(f"unknown dimension {dimension!r}; "
                         f"expected one of {sorted(_VOICEARENA_SCORE_CLOSING)}")
    opening = (_VOICEARENA_SCORE_OPENING_AUDIO if modality == "speech"
               else _VOICEARENA_SCORE_OPENING_TEXT)
    if scenario:
        opening += _VOICEARENA_SCENARIO.format(scenario=scenario)
    return (opening, _VOICEARENA_SCORE_CLOSING[dimension], VOICEARENA_SYSTEM)


# ── CANDOR prompts, reused for VoiceArena ────────────────────────────────────
#
# The CANDOR prompts above rate a conversation's *success* and never name the criterion,
# and the original VoiceArena pass scored that single number against both of VoiceArena's
# vote columns. The ablation keeps that as the `verbatim` reference and adds two variants
# in which exactly one clause -- the thing being rated -- is substituted, so each score is
# asked about the criterion it is then evaluated against.
#
# Everything else stays byte-identical to the original wording, including "Listen to this
# conversation" on text-modality runs: the original ran it that way on transcripts too,
# and changing it would confound the modality comparison with a prompt change.

CANDOR_PROMPT_NAMES = ("success", "CoT_summary", "CoT_questions")
CRITERION_VARIANTS = ("verbatim", "task_capability", "humanness")

# The two surface forms the success clause takes across the three prompts. Substitution is
# done on the rendered string so that whatever is not this clause is provably unchanged.
_SUCCESS_CLAUSES = (
    "how successful it is on a scale from 0 (not successful) to 10 (very successful)",
    "how successful the conversation is on a scale from 0 (not successful) to 10 (very successful)",
)

_CRITERION_CLAUSES = {
    "task_capability": ("how capably the agent handled the caller's task on a scale "
                        "from 0 (not capably) to 10 (very capably)"),
    "humanness": ("how natural and human-like the agent sounds on a scale "
                  "from 0 (not natural) to 10 (very natural)"),
}


def _substitute_criterion(prompt, variant):
    if variant == "verbatim":
        return prompt
    for clause in _SUCCESS_CLAUSES:
        if clause in prompt:
            return prompt.replace(clause, _CRITERION_CLAUSES[variant])
    raise ValueError(f"no success clause to substitute in prompt: {prompt[:60]!r}...")


def candor_voicearena_prompt(prompt_name, variant, target="the agent"):
    """One cell of the prompt x criterion grid.

    For CoT_questions the sub-question list is what carries the criterion: the verbatim
    variant keeps the CANDOR mood/liking questions (what was originally run on VoiceArena),
    while the criterion variants pour this file's VoiceArena sub-questions into the same
    scaffold.
    """
    if variant not in CRITERION_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {CRITERION_VARIANTS}")

    if prompt_name == "CoT_questions" and variant != "verbatim":
        questions = [question.format(target=target) for _, question in _COT_QUESTIONS[variant]]
        return _substitute_criterion(cot_questions_prompt(questions), variant)

    return _substitute_criterion(PROMPTS[prompt_name], variant)
