import evaluate
import torch

import torch.nn as nn
from tqdm import tqdm

from configs.config import settings

class EvalLSTM:
    def __init__(self, model, device, compute_rouge: bool = False):
        self.model = model
        self.device = device
        self.compute_rouge = compute_rouge
        self.criterion = nn.CrossEntropyLoss()


    def generate(self, input_ids: list[int], n_tokens: int, tokenizer) -> str:
        self.model.eval()
        gen_ids = self._generate_ids(input_ids, n_tokens)
        return tokenizer.decode(gen_ids)
    

    def _generate_ids(self, input_ids: list[int], n_tokens: int) -> list[int]:
        tokens = input_ids.copy()
        seq_len = len(input_ids)

        with torch.no_grad():
            for _ in range(n_tokens):
                x = torch.tensor(tokens[-seq_len:]).unsqueeze(0).to(self.device)
                logits = self.model(x)
                next_token = torch.argmax(logits[0, -1, :]).item()
                tokens.append(next_token) # type: ignore

        return tokens[seq_len:]  # только новые токены


    def generate_with_sampling(self, input_ids: list[int], n_tokens: int, tokenizer,
                               temperature: float = 1.0, top_k: int = 0) -> str:
        self.model.eval()
        tokens = input_ids.copy()
        seq_len = len(input_ids)

        with torch.no_grad():
            for _ in range(n_tokens):
                x = torch.tensor(tokens[-seq_len:]).unsqueeze(0).to(self.device)
                logits = self.model(x)[0, -1, :]   # (vocab_size,)

                # temperature — сглаживаем или заостряем распределение
                logits = logits / temperature

                # top-k — оставляем только k наиболее вероятных токенов
                if top_k > 0:
                    top_values, _ = torch.topk(logits, top_k)
                    logits[logits < top_values[-1]] = -float("inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token) # type: ignore

        generated = tokens[seq_len:]
        return tokenizer.decode(generated)


    def evaluate(self, loader, tokenizer=None, max_rouge_samples=50, max_gen_length=12):
        print("Start evaluate")
        self.model.eval()
        correct, total = 0, 0
        sum_loss = 0

        rouge = evaluate.load("rouge") if self.compute_rouge else None
        all_predictions = []
        all_references = []
        rouge_samples_count = 0 

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                x_output = self.model(x_batch)
                loss = self.criterion(x_output.transpose(1, 2), y_batch)
                preds = torch.argmax(x_output, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.numel()
                sum_loss += loss.item()
                
                if self.compute_rouge and tokenizer is not None and rouge_samples_count < max_rouge_samples:
                    for i in range(x_batch.size(0)):
                        if rouge_samples_count >= max_rouge_samples:
                            break
                            
                        tokens = x_batch[i].cpu().tolist()
                        target = y_batch[i].cpu().tolist()

                        prompt_len = max(1, int(len(tokens) * 0.75))
                        
                        # Ограничиваем длину генерации средним размером твита
                        n_generate = min(len(tokens) - prompt_len, max_gen_length)

                        gen_ids = self._generate_ids(tokens[:prompt_len], n_tokens=n_generate)
                        all_predictions.append(tokenizer.decode(gen_ids))
                        all_references.append(tokenizer.decode(target[prompt_len:]))
                        
                        rouge_samples_count += 1

        avg_loss = sum_loss / len(loader)
        accuracy = correct / total
        avg_rouge = None
        
        if self.compute_rouge and all_predictions:
            avg_rouge = rouge.compute(predictions=all_predictions, references=all_references) # type: ignore

        self.model.train()
        return avg_loss, accuracy, avg_rouge
