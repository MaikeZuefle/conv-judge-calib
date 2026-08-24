DEFAULT_SYSTEM = "You are a careful evaluator of conversations between people."


class Qwen35:
    """Qwen3.5-27B. 262k context, so full conversation transcripts fit without truncation.
    `thinking` switches on the model's native reasoning mode via the chat template."""

    MODEL_ID = "Qwen/Qwen3.5-27B"

    def __init__(self, thinking=False):
        import logging

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        logging.getLogger().setLevel(logging.ERROR)
        self.thinking = thinking

        try:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        except Exception:  # text-only checkpoints expose no processor
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt, content, modality="text", max_new_tokens=512,
                 preamble=None, system=None):
        import torch

        if modality != "text":
            raise ValueError("Qwen35 is text-only in this pipeline")

        parts = [p for p in (preamble, content, prompt) if p]
        conversation = [
            {"role": "system", "content": system or DEFAULT_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        text_out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return text_out

    # ── pointwise scoring via score-token logprobs ───────────────────────────

    def _build_inputs(self, prompt, content, modality="text"):
        if modality != "text":
            raise ValueError("Qwen35 is text-only")
        conversation = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": f"{content}\n\n{prompt}"},
        ]
        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
            enable_thinking=self.thinking,
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def _score_tokens(self, lo, hi):
        if hasattr(self, "_score_token_ids") and self._score_token_ids is not None:
            return self._score_token_ids
        ids, values, dropped = [], [], []
        for v in range(lo, hi + 1):
            enc = self.tokenizer.encode(str(v), add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0]); values.append(v)
            else:
                dropped.append(v)
        for v in dropped:
            first = self.tokenizer.encode(str(v), add_special_tokens=False)[0]
            if first not in ids:
                ids.append(first); values.append(v)
        if len(ids) < 2:
            raise RuntimeError(f"could not find distinct score tokens for {lo}-{hi}")
        self._score_token_ids = (ids, values)
        return self._score_token_ids

    def score(self, prompt, content, modality="text", lo=1, hi=10, **_):
        import torch

        ids, values = self._score_tokens(lo, hi)
        inputs = self._build_inputs(prompt, content, modality)
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0, -1, :].float()
        full = torch.softmax(logits, dim=-1)
        sel = full[torch.tensor(ids, device=full.device)]
        p_mass = float(sel.sum())
        if p_mass <= 0:
            return {"score": None, "argmax": None, "probs": {}, "p_mass": 0.0}
        norm = sel / sel.sum()
        vals_t = torch.tensor([float(v) for v in values], device=norm.device)
        expectation = float((norm * vals_t).sum())
        argmax_val = values[int(norm.argmax())]
        probs = {str(v): round(float(p), 6) for v, p in zip(values, norm.tolist())}
        torch.cuda.empty_cache()
        return {"score": expectation, "argmax": argmax_val, "probs": probs, "p_mass": p_mass}

    def score_with_cot(self, prompt, score_prompt, content, modality="text",
                       lo=1, hi=10, max_new_tokens=400, **_):
        import torch

        inputs = self._build_inputs(prompt, content, modality)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        reasoning = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        del inputs, generated
        torch.cuda.empty_cache()

        combined = f"{prompt}\n\n{reasoning}\n\n{score_prompt}"
        res = self.score(combined, content, modality=modality, lo=lo, hi=hi)
        res["reasoning"] = reasoning
        return res
