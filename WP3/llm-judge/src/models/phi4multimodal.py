MODEL_ID = "microsoft/Phi-4-multimodal-instruct"


class Phi4Multimodal:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(MODEL_ID)

    def generate(self, prompt, content, modality="speech", max_new_tokens=512):
        import soundfile as sf
        import torch

        if modality == "speech":
            audio, samplerate = sf.read(content)
            text = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
            inputs = self.processor(text=text, audios=[(audio, samplerate)], return_tensors="pt")
        else:
            text = f"<|user|>{content}\n\n{prompt}<|end|><|assistant|>"
            inputs = self.processor(text=text, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, generation_config=self.generation_config
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.cuda.empty_cache()
        return output[0]
