import math
import torch

import torch.nn as nn
from tqdm import tqdm

from configs.config import settings

class LSTMTrain:
    def __init__(self, model, device, evaluator):
        self.model = model
        self.device = device
        self.evaluator = evaluator
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()


    def model_train(self, train_loader, val_loader, tokenizer):
        print("Start model train")
        n_epochs = settings.EPOCH
        metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_ppl': []
        }
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                x_output = self.model(x_batch)
                loss = self.criterion(x_output.transpose(1, 2), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            val_loss, val_acc, avg_rouge = self.evaluator.evaluate(val_loader, tokenizer)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            metrics_history['epoch'].append(epoch+1)
            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_accuracy'].append(val_acc)
            metrics_history['val_ppl'].append(val_ppl)
            rouge_str = ""
            if avg_rouge:
                rouge_str = (f"| ROUGE-1: {avg_rouge['rouge1']:.4f} "
                             f"| ROUGE-2: {avg_rouge['rouge2']:.4f} "
                             f"| ROUGE-L: {avg_rouge['rougeL']:.4f} "
                             f"| ROUGE-Lsum: {avg_rouge['rougeLsum']:.4f}")
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} "
                f"| Val Loss: {val_loss:.3f} | Val PPL: {val_ppl:.2f} | Val Accuracy: {val_acc:.2%}\n{rouge_str}")
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
                }
            torch.save(checkpoint, f'models/checkpoint-{epoch+1}-loss-{train_loss}.pth')
        return metrics_history