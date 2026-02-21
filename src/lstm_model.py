import torch

import torch.nn as nn

from configs.config import settings
from src.next_token_dataset import tokenizer


class LSTMRNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=settings.EMBEDDING_DIM,
                 hidden_dim=settings.HIDDEN_DIM,
                 num_layers=settings.NUM_LAYERS,
                 dropout=settings.DROPOUT
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, #  должно совпадать с embedding_dim
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

    def forward(self, x):
        emb = self.embedding(x)
        dout = self.dropout(emb)
        out, _ = self.rnn(dout)
        out = self.norm(out)
        logits = self.fc(out)
        return logits
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vocab_size = tokenizer.vocab_size
model = LSTMRNNClassifier(vocab_size=vocab_size).to(device)
