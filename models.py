import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLineBLSTM(nn.Module):
    '''
    Baseline Bi-directional LSTM Model to check what we expect to see when using a Bi-directional LSTM
    Inputs:
        vocab_size (int): size of vocabulary
        embedding_dim (int): size of embedding layer
        hidden_dim (int): size of hidden_dim for LSTM
        n_layers (int): number of LSTM to stack
        tagset_size (int): size of output space (number of tags)
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, tagset_size):
        super(BaseLineBLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, bidirectional=True)
        self.hiddentotag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        embeds = self.word_embed(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hiddentotag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores