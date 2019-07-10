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

def perplexity(preds, labels):
    """Compute average perplexity of a batch.

    Args:
        preds of shape (seq_len, vocab_size, batch_size)
        labels of (seq_len, batch_size)
    """
    # deactivate grad_fn tracking history
    preds = preds.detach()
    labels = labels.detach()

    n = labels.shape[0]

    labels = labels.unsqueeze(1)    # (seq_len, 1, batch_size)
    probs = preds.gather(1, labels) # (seq_len, 1, batch_size)

    # OPERATIONS ORDER CAN'T BE CHANGED!!
    # IF perplex.prod(0).prod(0) EVALUATED FIRST, IT'LL RESULT TO 0
    perplex = 1/probs
    perplex = perplex.type(torch.float).pow(1/n)
    perplex = perplex.prod(0).prod(0)
    return perplex.mean()