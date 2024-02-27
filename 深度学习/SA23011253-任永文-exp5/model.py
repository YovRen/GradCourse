from turtle import forward
import torch
import torch.nn as nn
import math
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:,:x.size(1 )])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query_fc = nn.Linear(d_model, d_model)
        self.key_fc = nn.Linear(d_model, d_model)
        self.value_fc = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Linear transformations & Reshape for multihead
        query = self.query_fc(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        key = self.key_fc(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        value = self.value_fc(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0,-1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        # Reshape and concatenate
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.head_dim)
        # Final linear layer
        output = self.fc(attention_output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x))) 


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead,dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ffn,dropout)
        self.sublayer1 = SublayerConnection(d_model,dropout)
        self.sublayer2 = SublayerConnection(d_model,dropout)

    def forward(self, x, src_mask):
        x = self.sublayer1(x,lambda x:self.self_attn(x,x,x,src_mask))
        x = self.sublayer2(x,self.feed_forward)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead,dropout)
        self.src_attn = MultiHeadAttention(d_model, nhead,dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ffn,dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer1(x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer2(x,lambda x:self.src_attn(x,memory,memory,src_mask))
        x = self.sublayer2(x,self.feed_forward)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,d_ffn, nhead, dropout, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model,padding_idx=2)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model,padding_idx=2)
        self.positional_encoding = PositionalEncoding(d_model, dropout,max_len=72)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model,d_ffn, nhead, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model,d_ffn, nhead, dropout) for _ in range(num_decoder_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_mask = (src != 2).unsqueeze(-2)
        tgt_mask = (tgt != 2).unsqueeze(-2) & torch.from_numpy(1-np.triu(np.ones((1,tgt.size(-1),tgt.size(-1))),k=1).astype('uint8')).type_as(tgt != 0)
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        memory = self.norm(src)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        output = self.norm(tgt)
        output = torch.log_softmax(self.fc(output),dim=-1)
        return output

