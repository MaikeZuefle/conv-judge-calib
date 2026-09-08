MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs. Only return the answer requested. Do not include "
    "any explanation or introductions."
)


class Qwen3Omni:
    def __init__(self):
        import logging

        import torch
        from transformers import (
            BitsAndBytesConfig,
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        logging.getLogger().setLevel(logging.ERROR)

        # 4-bit quantized to fit the same single-GPU budget as the other (much smaller) models
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            enable_audio_output=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)

        # The installed transformers version always projects every prefill position through
        # lm_head (~150k-vocab), even though generate() only ever samples from the last one.
        # For CANDOR's long audio inputs that single matmul can require tens of GiB and OOM.
        # Only the last position is ever needed for sampling, so slice before projecting.
        lm_head = self.model.thinker.lm_head
        original_lm_head_forward = lm_head.forward
        lm_head.forward = lambda hidden_states: original_lm_head_forward(hidden_states[:, -1:, :])

    def generate(self, prompt, content, modality="speech", max_new_tokens=768):
        import torch
        from qwen_omni_utils import process_mm_info

        if modality == "speech":
            user_content = [{"type": "audio", "audio": content}, {"type": "text", "text": prompt}]
        else:
            user_content = [{"type": "text", "text": f"{content}\n\n{prompt}"}]

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            truncation=False,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        text_ids, _audio = self.model.generate(
            **inputs,
            speaker="Ethan",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=False,
            max_new_tokens=max_new_tokens,
        )
        output = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        torch.cuda.empty_cache()
        return output[0]
