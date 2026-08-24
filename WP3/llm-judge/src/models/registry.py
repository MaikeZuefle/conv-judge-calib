"""The one place a judge model name maps to the class that implements it.

The runners previously each carried their own `MODELS` dict. They disagreed, and the
disagreements were silent: run_voicearena_score_judge offered qwen25omni for pointwise
scoring, but qwen25omni implements only `generate`, so choosing it failed at scoring
time with an AttributeError rather than at argument parsing.

Models are not interchangeable, so the subsets below name the capability a runner
actually needs. Import the subset that matches what your runner calls:

  ALL       -- anything with .generate(); pairwise/BWS runners
  SCORING   -- also has .score() and .score_with_cot(); pointwise runners
  SPEECH    -- accepts modality="speech"

Values are factories rather than classes so a configured variant (qwen35_thinking) can
sit alongside a plain class. Construct with `ALL[name]()`.
"""

from models.avflamingo import AVFlamingo
from models.llama33 import Llama33
from models.phi4multimodal import Phi4Multimodal
from models.prometheus import Prometheus8x7B
from models.qwen25omni import Qwen25Omni
from models.qwen35 import Qwen35

ALL = {
    "qwen25omni": Qwen25Omni,
    "qwen35": Qwen35,
    "qwen35_thinking": lambda: Qwen35(thinking=True),
    "prometheus8x7b": Prometheus8x7B,
    "llama33": Llama33,
    "phi4multimodal": Phi4Multimodal,
    "avflamingo": AVFlamingo,
}

# Implement score()/score_with_cot() *meaningfully*, i.e. read a rating from the
# probability distribution over the score tokens instead of from generated text.
#
# qwen35_thinking is deliberately absent even though it is a Qwen35 and so inherits
# score(). score() reads the logits at the first generated position; with thinking
# enabled the chat template has the model open a reasoning block there, so essentially
# no probability mass lands on the score tokens and score() returns None. Presence of
# the method is not the same as the method being usable.
SCORING = {name: ALL[name] for name in ("phi4multimodal", "qwen35")}

# Configured variants whose class implements score() but for which it is not meaningful.
SCORING_EXCLUDED = {"qwen35_thinking": "thinking mode emits reasoning at position 0"}

# Accept modality="speech". The rest are text-only and raise on audio input.
SPEECH = {name: ALL[name] for name in ("qwen25omni", "avflamingo", "phi4multimodal")}
