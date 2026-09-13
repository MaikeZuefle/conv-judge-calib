SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech. Only "
    "return the answer requested. Do not include any explanation or introductions."
)


class Qwen25Omni:
    def __init__(self):
        import logging

        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        logging.getLogger().setLevel(logging.ERROR)

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    def generate(self, prompt, content, modality="speech", max_new_tokens=512,
                 preamble=None, system=None, clip_prefix="CONVERSATION"):
        """`preamble` is placed before the content, so a task can be stated up front
        rather than only after a long transcript. `system` overrides the default system
        prompt, which forbids explanation and so suppresses chain-of-thought styles.
        `clip_prefix` sets the label inserted before each audio clip, e.g. "CALL"."""
        import torch
        from qwen_omni_utils import process_mm_info

        if modality == "speech":
            # content is one audio path or several; several are interleaved with
            # "[{clip_prefix} n]" labels so the model can refer to them by number
            clips = content if isinstance(content, (list, tuple)) else [content]
            user_content = [{"type": "text", "text": preamble}] if preamble else []
            for i, clip in enumerate(clips, 1):
                if len(clips) > 1:
                    user_content.append({"type": "text", "text": f"[{clip_prefix} {i}]"})
                user_content.append({"type": "audio", "audio": clip})
            user_content.append({"type": "text", "text": prompt})
        else:
            parts = [p for p in (preamble, content, prompt) if p]
            user_content = [{"type": "text", "text": "\n\n".join(parts)}]

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system or SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        text_ids = self.model.generate(**inputs, return_audio=False, max_new_tokens=max_new_tokens)
        generated_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.cuda.empty_cache()
        return output[0]
