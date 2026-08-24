MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# Phi-4-multimodal's remote code calls prepare_inputs_for_generation on the base model
# class, which moved to GenerationMixin in transformers >= 4.50. Run this in the pinned
# env built by scripts/build_phi4_env.sh (transformers 4.48.2), not the shared judge env.


class Phi4Multimodal:
    def __init__(self, attn_implementation=None):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        if attn_implementation is None:
            try:
                import flash_attn  # noqa: F401
                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = "eager"

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation=attn_implementation,
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
        self._score_token_ids = None

    # ── prompt construction ──────────────────────────────────────────────────

    def _build_inputs(self, prompt, content, modality):
        import soundfile as sf

        if modality == "speech":
            clips = content if isinstance(content, (list, tuple)) else [content]
            audios = [sf.read(c) for c in clips]
            audio_tags = "".join(f"<|audio_{i + 1}|>" for i in range(len(clips)))
            text = f"<|user|>{audio_tags}{prompt}<|end|><|assistant|>"
            inputs = self.processor(text=text, audios=audios, return_tensors="pt")
        else:
            text = f"<|user|>{content}\n\n{prompt}<|end|><|assistant|>"
            inputs = self.processor(text=text, return_tensors="pt")
        return inputs.to(self.model.device)

    # ── free-form generation (pairwise judging) ──────────────────────────────

    def generate(self, prompt, content, modality="speech", max_new_tokens=512, **_):
        import torch

        inputs = self._build_inputs(prompt, content, modality)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, generation_config=self.generation_config
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.cuda.empty_cache()
        return output[0]

    # ── pointwise scoring via score-token logprobs ───────────────────────────

    def _score_tokens(self, lo, hi):
        """Token id for each integer score in [lo, hi].

        Only scores whose surface form is a SINGLE token are usable: the expectation is
        read off one forward pass at one position, so a score spanning several tokens has
        no probability at that position. For 1-10 this loses nothing on a BPE tokenizer
        with digit tokens -- "10" is the only two-token case and is handled by scoring
        its first token, which is unambiguous here because no other option starts with 1
        except 1 itself. That ambiguity is why 1-100 is a bad fit for this method.
        """
        if self._score_token_ids is not None:
            return self._score_token_ids

        tok = self.processor.tokenizer
        ids, values, dropped = [], [], []
        for v in range(lo, hi + 1):
            enc = tok.encode(str(v), add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
                values.append(v)
            else:
                dropped.append(v)
        if dropped:
            # Fall back to the leading token, which for "10" vs "1" differs on most BPE
            # vocabularies; if it collides we would double-count, so verify uniqueness.
            for v in dropped:
                first = tok.encode(str(v), add_special_tokens=False)[0]
                if first not in ids:
                    ids.append(first)
                    values.append(v)
        if len(ids) < 2:
            raise RuntimeError(f"could not find distinct score tokens for {lo}-{hi}")
        self._score_token_ids = (ids, values)
        return self._score_token_ids

    def score_with_cot(self, prompt, score_prompt, content, modality="speech",
                       lo=1, hi=10, max_new_tokens=400, **_):
        """Answer sub-questions first, then score conditioned on those answers.

        Two passes over the same audio: the first generates the reasoning, the second
        reads the score distribution with the reasoning already in context. Keeping the
        score elicitation separate means it is still a single token whose distribution
        can be read directly -- parsing a number out of generated JSON would hand back
        the integer ties that pointwise scoring exists to avoid.

        Returns the same fields as score(), plus "reasoning".
        """
        import torch

        inputs = self._build_inputs(prompt, content, modality)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generated = generated[:, inputs["input_ids"].shape[1]:]
        reasoning = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        del inputs, generated
        torch.cuda.empty_cache()

        # Rebuild as one turn: question block, the model's own answers, then the rating
        # request. Phi-4's chat format has no multi-turn audio, so the reasoning is
        # folded back into a single user/assistant exchange.
        combined = f"{prompt}\n\n{reasoning}\n\n{score_prompt}"
        res = self.score(combined, content, modality=modality, lo=lo, hi=hi)
        res["reasoning"] = reasoning
        return res

    def score(self, prompt, content, modality="speech", lo=1, hi=10, **_):
        """Return a continuous score in [lo, hi] as the expectation over score tokens.

        A greedy integer would tie a large fraction of items -- 1-10 judging collapses
        onto a couple of values in practice, and tied items cannot be ranked against each
        other at all. Taking sum(p(k) * k) over the score tokens at the first generated
        position keeps the model's own uncertainty as resolution, so every item gets a
        distinct value and near-ties stay visibly near-tied.

        Returns {"score": float, "argmax": int, "probs": {value: prob}, "p_mass": float}.
        `p_mass` is how much probability sat on valid score tokens at all -- if it is low
        the model was not answering the question and the score should not be trusted.
        """
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
            torch.cuda.empty_cache()
            return {"score": None, "argmax": None, "probs": {}, "p_mass": 0.0}

        norm = sel / sel.sum()
        expectation = float((norm * torch.tensor(
            [float(v) for v in values], device=norm.device)).sum())
        probs = {int(v): float(p) for v, p in zip(values, norm)}
        argmax = int(values[int(torch.argmax(norm))])

        torch.cuda.empty_cache()
        return {"score": expectation, "argmax": argmax, "probs": probs, "p_mass": p_mass}
