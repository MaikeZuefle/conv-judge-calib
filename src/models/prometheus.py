"""Prometheus 2 (8x7B), an LLM trained specifically for evaluation.

Its relative-grading mode compares two responses and answers "A" or "B", which maps
directly onto pairwise conversation comparison. Note the model is Mixtral-based with a
32k context and was trained on evaluation instances of a few hundred to ~2k tokens, so
9k-token conversation transcripts are well outside its training distribution.
"""

from models.hf_causal import HFCausalLM

SYSTEM_PROMPT = (
    "You are a fair judge assistant assigned to deliver insightful feedback that compares "
    "individual performances, highlighting how each stands relative to others within the "
    "same cohort."
)

# Official relative-grading template, reference-free variant.
RELATIVE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Score Rubric:
{rubric}

###Feedback: """

INSTRUCTION = (
    "Two people who had never met before were asked to talk with each other for a few "
    "minutes and get to know one another. Below is a transcript of how each pair's "
    "conversation went."
)

RUBRIC = (
    "Did the participants themselves enjoy the conversation? A better conversation is one "
    "that its own participants found more enjoyable: they were more engaged with each "
    "other, warmer, got on better, and came away more positive about the exchange."
)


class Prometheus8x7B(HFCausalLM):
    MODEL_ID = "prometheus-eval/prometheus-8x7b-v2.0"

    def build_prompt(self, transcripts):
        if len(transcripts) != 2:
            raise ValueError("Prometheus relative grading compares exactly two responses")
        return RELATIVE_PROMPT.format(
            instruction=INSTRUCTION, response_a=transcripts[0],
            response_b=transcripts[1], rubric=RUBRIC,
        )

    def generate(self, prompt, content, modality="text", max_new_tokens=512,
                 preamble=None, system=None):
        """`content` is the fully built relative-grading prompt; the batch runner
        constructs it via build_prompt so the template stays with the model."""
        conversation = [
            {"role": "user", "content": f"{system or SYSTEM_PROMPT}\n\n{content}"},
        ]
        return self._generate_chat(conversation, max_new_tokens)
