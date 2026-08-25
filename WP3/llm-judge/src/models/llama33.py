from models.hf_causal import HFCausalLM

DEFAULT_SYSTEM = "You are a careful evaluator of conversations between people."


class Llama33(HFCausalLM):
    """Llama-3.3-70B-Instruct. Text-only, 128k context, ~140 GB in bf16 so it needs at
    least two A100s. Included as a third lineage alongside Qwen and Prometheus, so that a
    difference between model generations can be told apart from a difference between
    model families."""

    MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

    def generate(self, prompt, content, modality="text", max_new_tokens=512,
                 preamble=None, system=None):
        if modality != "text":
            raise ValueError("Llama33 is text-only")

        parts = [p for p in (preamble, content, prompt) if p]
        conversation = [
            {"role": "system", "content": system or DEFAULT_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        return self._generate_chat(conversation, max_new_tokens)
