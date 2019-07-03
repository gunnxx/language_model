import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 lstm_hidden_dim: int,
                 fc_hidden_dim: int,
                 vocab_size: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 ) -> None:

        super(Net, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, vocab_size)

    def forward(self, s):        # (seq_len, batch_size)
        s = self.embedding(s)    # (seq_len, batch_size, embedding_dim)
        s, _ = self.lstm(s)      # (seq_len, batch_size, lstm_hidden_dim)
        s = s.contiguous()
        s = self.fc1(s)          # (seq_len, batch_size, fc_hidden_dim)
        s = F.relu(s)
        s = self.fc2(s)          # (seq_len, batch_size, vocab_size)
        s = F.softmax(s, dim=2)
        s = s.transpose(1, 2)    # (seq_len, vocab_size, batch_size)
        return s