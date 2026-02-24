import evaluate
import torch

import torch.nn as nn

from configs.config import settings


class EvalLSTM:
    def __init__(self, model, device, compute_rouge: bool = False):
        self.model = model
        self.device = device
        self.compute_rouge = compute_rouge
        self.criterion = nn.CrossEntropyLoss()


    def evaluate(self, loader, tokenizer=None, max_rouge_samples=settings.MAX_ROUGE_SAMPLES, max_gen_length=settings.MAX_GEN_LENGTH):
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
                        prompt_len = max(1, int(len(tokens) * 0.75))
                        n_generate = min(len(tokens) - prompt_len, max_gen_length)

                        if n_generate <= 0:
                            continue

                        prompt_ids = tokens[:prompt_len]
                        gen_ids = self.model._generate_ids(prompt_ids, n_tokens=n_generate)
                        # продолжение исходного X после промпта
                        ref_ids = tokens[prompt_len:prompt_len + n_generate]
                        pred_full = tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=True)
                        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                        pred_cont = pred_full[len(prompt_text):].strip()
                        ref_cont = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()

                        all_predictions.append(pred_cont)
                        all_references.append(ref_cont)
                        
                        rouge_samples_count += 1

        avg_loss = sum_loss / len(loader)
        accuracy = correct / total
        avg_rouge = None
        print(f"all_predictions ---> {all_predictions}")
        print(f"all_references ---> {all_references}")
        if self.compute_rouge and all_predictions:
            avg_rouge = rouge.compute(predictions=all_predictions, references=all_references) # type: ignore

        self.model.train()
        return avg_loss, accuracy, avg_rouge
