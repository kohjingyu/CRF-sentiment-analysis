import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import XLNetModel, XLNetConfig

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
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, bidirectional=True)
        self.hiddentotag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        embeds = self.word_embed(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hiddentotag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)
        return tag_scores


class Seq2Seq(nn.Module):
    '''
    Sequence to Sequence Model to capture context of sentence before assigning tags, uses LSTM
    Inputs:
        vocab_size (int): size of vocabulary
        embedding_dim (int): size of embedding layer
        hidden_dim (int): size of hidden_dim for LSTM
        n_layers (int): number of LSTM to stack
        tagset_size (int): size of output space (number of tags)
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, tagset_size):
        super(Seq2Seq, self).__init__()

        self.encoder_embed = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               n_layers, bidirectional=True)

        self.decoder_embed = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim,
                               n_layers*2)

        self.hiddentotag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        encode_embed = self.encoder_embed(x)
        _, (hn, cn) = self.encoder(encode_embed)

        decode_embed = self.decoder_embed(x)
        lstm_out, _ = self.decoder(decode_embed, (hn, cn))

        tag_space = self.hiddentotag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)
        return tag_scores


class AttentionSeq2Seq(nn.Module):
    '''
    Sequence to Sequence Model with some attention to capture context of sentence before assigning tags, uses LSTM
    Inputs:
        vocab_size (int): size of vocabulary
        embedding_dim (int): size of embedding layer
        hidden_dim (int): size of hidden_dim for LSTM
        n_layers (int): number of LSTM to stack
        tagset_size (int): size of output space (number of tags)
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, tagset_size):
        super(AttentionSeq2Seq, self).__init__()

        self.encoder_embed = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               n_layers, bidirectional=True)

        self.decoder_embed = nn.Embedding(vocab_size, embedding_dim)

        self.attention = nn.Linear(embedding_dim+hidden_dim*2, embedding_dim)

        self.decoder = nn.LSTM(embedding_dim,
                               hidden_dim, n_layers, bidirectional=True)

        self.hiddentotag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        embeds = self.encoder_embed(x)
        encoder_out, (hn, cn) = self.encoder(embeds)

        decode_emb = self.decoder_embed(x)
        attn = self.attention(torch.cat([encoder_out, decode_emb], dim=2))
        attn = F.softmax(attn, dim=2)

        decoder_out, _ = self.decoder(decode_emb, (hn, cn))

        tag_space = self.hiddentotag(decoder_out)
        tag_scores = F.softmax(tag_space, dim=2)

        return tag_scores

class XLNetTest(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, tagset_size):
        super(XLNetTest, self).__init__()
        config = XLNetConfig.from_pretrained('xlnet-large-cased')
        self.model = XLNetModel(config) 

        # self.decoder = nn.LSTM(1024, hidden_dim,
        #                        n_layers)

        self.hiddentotag = nn.Linear(1024, tagset_size)

    def forward(self, x):
        # No grad the XLNet
        with torch.no_grad():
            outputs = self.model(x)[0]

        # lstm_out, _ = self.decoder(outputs)

        tag_space = self.hiddentotag(outputs)
        tag_scores = F.softmax(tag_space, dim=2)

        return tag_scores

class XLNetLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, tagset_size):
        super(XLNetLSTM, self).__init__()
        config = XLNetConfig.from_pretrained('xlnet-large-cased')
        self.model = XLNetModel(config) 

        self.decoder = nn.LSTM(1024, hidden_dim,
                               n_layers)

        self.hiddentotag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        # No grad the XLNet
        with torch.no_grad():
            outputs = self.model(x)[0]

        lstm_out, _ = self.decoder(outputs)

        tag_space = self.hiddentotag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)

        return tag_scores