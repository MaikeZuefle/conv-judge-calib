MODEL_ID = "Qwen/Qwen3.6-27B"


class Qwen36_27B:
    """Text-only. Qwen3.6-27B has no audio input, so this only supports modality="text"
    (a transcript, or a summary produced by another model)."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")

    def generate(self, prompt, content, modality="text", max_new_tokens=1024):
        import torch

        if modality != "text":
            raise ValueError("Qwen36_27B only supports modality='text'")

        messages = [{"role": "user", "content": f"{content}\n\n{prompt}"}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return output[0]
