import torch

import torch.nn as nn
from src.next_token_dataset import NextTokenDataset, tokenizer

# class LSTMRNNClassifier(nn.Module):
#     def __init__(self, vocab_size, input_dim=3, hidden_dim=128):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
#         self.fc = nn.Linear(hidden_dim, vocab_size)


#     def forward(self, x):
#         emb = self.embedding(x)
#         out, _ = self.rnn(emb)
#         linear_out = self.fc(out)
#         return linear_out


class LSTMRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=1, dropout=0.3):
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
        emb = self.embedding(x)   # (batch, seq_len, embedding_dim)
        dout = self.dropout(emb)  
        out, _ = self.rnn(dout)   # (batch, seq_len, hidden_dim)
        out = self.norm(out)
        logits = self.fc(out)     # (batch, seq_len, vocab_size)
        return logits    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vocab_size = tokenizer.vocab_size
# input_dim = 128
# hidden_dim = 128
model = LSTMRNNClassifier(vocab_size=vocab_size).to(device)


# # Сравнение
# print(f"{'RNN Type':<8} | {'Combine':<6} | {'Params':>10}")
# print("-" * 35)
# model = LSTMRNNClassifier(vocab_size, hidden_dim)
# param_count = count_parameters(model)
# print(f"{param_count:>10,}")