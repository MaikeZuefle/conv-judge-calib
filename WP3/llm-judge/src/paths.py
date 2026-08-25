"""Every filesystem location this package reads, in one place.

Previously each module hardcoded its own absolute path. That made the code
Skipjack- and user-specific in ways that were invisible until something failed:
the same CANDOR directory was reached through two different prefixes (one of them
via a symlink in a single user's home), and one path pointed into another user's
home directory entirely.

Each root can be overridden by an environment variable; the defaults are the
literals the scripts used before, so existing invocations behave identically.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_path(var, default):
    return Path(os.environ.get(var, default))


# Corpora committed to / staged in the repo itself.
DATA_ROOT = _env_path("JSALT_DATA_ROOT", REPO_ROOT / "data")
VOICEARENA_ROOT = DATA_ROOT / "Human-AI-interaction" / "Human-Agent-VoiceArena"

# Shared project scratch: source corpora too large for the repo, plus derived caches.
SCRATCH_ROOT = _env_path("JSALT_SCRATCH_ROOT", "/weka/scratch/jhu/jsalt2026-lgarci27/simulator")

# Sits beside VoiceArena under Human-AI-interaction, which in this checkout is a symlink
# to the shared scratch copy. The previous literal omitted the Human-AI-interaction level
# and so pointed at a directory that has never existed -- humanai_data could not load at
# all. plot_full_comparison.py already used the correct path.
HUMANAI_ROOT = _env_path(
    "JSALT_HUMANAI_ROOT",
    DATA_ROOT / "Human-AI-interaction" / "Human-AI-Natural-Conversation-Recording",
)
CANDOR_ROOT = _env_path("JSALT_CANDOR_ROOT", SCRATCH_ROOT / "data" / "CANDOR")
CANDOR_SURVEYS_RAW = CANDOR_ROOT / "all_surveys_raw.csv"
CANDOR_SURVEYS_BY_PARTICIPANT = CANDOR_ROOT / "surveys_agg_by_participant.csv"
VOICEARENA_VOTES = (SCRATCH_ROOT / "data" / "Human-AI-interaction"
                    / "Human-Agent-VoiceArena" / "vote_log.csv")

# Derived caches — regenerable, so they live on scratch rather than in the repo.
VOICEARENA_ASR_CACHE = SCRATCH_ROOT / "voicearena_asr"
HUMANAI_MIX_CACHE = SCRATCH_ROOT / "humanai_mixed"

# Second ASR source for VoiceArena, to ablate transcript quality against Whisper.
# Produced with nvidia/parakeet-tdt-0.6b-v3 and shipped as a zip in the repo; unpacked
# into the cache on first use.
VOICEARENA_PARAKEET_ZIP = DATA_ROOT / "transcripts_voice_arena_parakeet.zip"
VOICEARENA_PARAKEET_CACHE = SCRATCH_ROOT / "voicearena_asr_parakeet"

# The annotation conversation pairs are not in this repo; they were produced in a
# collaborator's checkout. Set JSALT_CONV_PAIRS to point at your own copy.
CONV_PAIRS_DIR = _env_path(
    "JSALT_CONV_PAIRS",
    "/home/jhu/jsalt2026-ext-mmuller/workspace/Conv-AI-Sim-JSALT2026/"
    "WP3/llm-judge/annotation_split/conv_pairs",
)
