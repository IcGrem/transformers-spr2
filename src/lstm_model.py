import torch

import torch.nn as nn

from configs.config import settings
from src.next_token_dataset import tokenizer


class LSTMRNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 device,
                 embedding_dim=settings.EMBEDDING_DIM,
                 hidden_dim=settings.HIDDEN_DIM,
                 num_layers=settings.NUM_LAYERS,
                 dropout=settings.DROPOUT
                 ):
        super(LSTMRNNClassifier, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, #  input_size должно совпадать с embedding_dim
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0 #  пока оставить
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, hidden: tuple|None = None):
        emb = self.embedding(x)
        dout = self.dropout(emb)
        out, hidden = self.rnn(dout, hidden)
        out = self.norm(out)
        logits = self.fc(out)
        return logits, hidden

    def generate(self, input_ids: list[int], n_tokens: int, tokenizer, sampling: bool = False) -> str:
        self.eval()
        if sampling:
            gen_ids = self._generate_with_sampling(input_ids, n_tokens)
        else:
            gen_ids = self._generate_ids(input_ids, n_tokens)
        return tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    def _generate_ids(self, input_ids: list[int], n_tokens: int) -> list[int]:
        tokens = input_ids.copy()
        hidden = None
        with torch.no_grad():
            x_init = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            _, hidden = self(x_init, hidden)
            for _ in range(n_tokens):
                # Берем только последний сгенерированный токен
                x = torch.tensor([[tokens[-1]]], dtype=torch.long, device=self.device)
                # Передаем hidden из предыдущего шага
                logits, hidden = self(x, hidden)
                next_token = torch.argmax(logits[0, -1, :]).item()
                tokens.append(next_token) # type: ignore
        return tokens[len(input_ids):]

    def _generate_with_sampling(self, input_ids: list[int], n_tokens: int,
                                temperature: float = settings.TEMPERATURE, 
                                top_k: int = settings.TOP_K) -> list[int]:
        tokens = input_ids.copy()
        hidden = None
        with torch.no_grad():
            x_init = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            _, hidden = self(x_init, hidden)
            for _ in range(n_tokens):
                x = torch.tensor([[tokens[-1]]], dtype=torch.long, device=self.device)
                logits, hidden = self(x, hidden)
                logits = logits[0, -1, :]
                logits = logits / temperature
                if top_k > 0:
                    top_values, _ = torch.topk(logits, top_k)
                    logits[logits < top_values[-1]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token) # type: ignore
        return tokens[len(input_ids):]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vocab_size = tokenizer.vocab_size
model = LSTMRNNClassifier(vocab_size=vocab_size, device=device).to(device)
