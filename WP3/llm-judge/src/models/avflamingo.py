MODEL_ID = "nvidia/nemotron-labs-audio-visual-flamingo-hf"


class AVFlamingo:
    def __init__(self):
        import torch
        from transformers import AudioVisualFlamingoForConditionalGeneration, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", dtype=dtype, load_audio_in_video=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, load_audio_in_video=True, num_video_frames=128)

    def generate(self, prompt, content, modality="speech", max_new_tokens=512):
        if modality == "speech":
            user_content = [{"type": "audio", "path": content}, {"type": "text", "text": prompt}]
        else:
            user_content = [{"type": "text", "text": f"{content}\n\n{prompt}"}]

        conversation = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return output[0]
