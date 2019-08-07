'''
Originally from http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout, n_block):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoder(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout, n_block):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(d_model)
        self.layer_cache = {}
        self.n_block = n_block

    def init_cache(self):
        for i in range(self.n_block):
            self.layer_cache[i] = None
    
    def update_cache(self, key, value):
        if self.layer_cache[key] is None:
            self.layer_cache[key] = value
        else:
            self.layer_cache[key] = torch.cat([self.layer_cache[key], value], dim=1)

    def forward(self, x, m, step_wise=False):
        for i, layer in enumerate(self.layers):
            if step_wise:
                self.update_cache(i, x)
                x = layer(x, m, self.layer_cache[i])
            else:
                x = layer(x, m)
        outs, attns = self.norm(x), []
        for layer in self.layers:
            attns.append(layer.dec_attn.attn.sum(1))
        attns = torch.stack(attns, dim=0).sum(0)
        outputs = {'outs': outs, 'attns': attns}
        return outputs


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.dec_attn = MultiHeadedAttention(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_m, dec_m=None):
        mask = subsequent_mask(x.size(0), x.size(1))
        if dec_m is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, dec_m, dec_m))
        x = self.sublayer[1](x, lambda x: self.dec_attn(x, enc_m, enc_m))
        return self.sublayer[2](x, self.feed_forward)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False).cuda()
        return self.dropout(x)


class PositionalEmb(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEmb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = x + self.pe(Variable(torch.range(1,x.size(1))).long().cuda()).unsqueeze(0)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def subsequent_mask(batch, size):
    "Mask out subsequent positions."
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = (torch.from_numpy(subsequent_mask) == 0)
    mask = mask.cuda()
    return mask