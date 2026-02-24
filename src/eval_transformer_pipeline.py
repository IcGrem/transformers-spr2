import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM


class EvalTransformer:
    def __init__(self, model_name: str, device: str|None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Используемое устройство: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.rouge_metric = evaluate.load("rouge")


    def generate_texts(self, prompts: list[str], max_length: int = 10, num_return_sequences: int = 1, **generation_kwargs):
        generated_texts = []
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )
        for i in range(output_ids.shape[0]):
            text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts


    def calculate_rouge(self, predictions: list[str], references: list[str]):
        results = self.rouge_metric.compute(
            predictions=predictions, 
            references=references
        )
        return results # type: ignore


    def validate(self, prompts: list[str], references: list[str], generation_config: dict):
        print("Начало генерации текстов")
        predictions = self.generate_texts(prompts, **generation_config)
        print("Вычисление метрик ROUGE")
        metrics = self.calculate_rouge(predictions, references)
        return metrics
