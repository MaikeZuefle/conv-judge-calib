"""Shared loading and greedy decoding for the text-only causal-LM judges.

Llama33 and Prometheus8x7B differ only in how they lay out the chat turns; everything
around that -- bf16 sharded loading, applying the chat template, greedy decode, slicing
the prompt back off, freeing the cache -- is the same, so it lives here.
"""


class HFCausalLM:
    """bf16, device_map="auto", greedy decoding. Subclasses set MODEL_ID and build the
    conversation their prompt format expects, then call _generate_chat."""

    MODEL_ID = None

    def __init__(self):
        import logging

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logging.getLogger().setLevel(logging.ERROR)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def _generate_chat(self, conversation, max_new_tokens):
        import torch

        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        text_out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return text_out
