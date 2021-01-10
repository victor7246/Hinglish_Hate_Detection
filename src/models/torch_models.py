from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .torch_layers import Encoder, TimeDistributed, EncoderWithoutEmbedding, Embedder, PositionalEncoding, PositionalEncoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, d_model, max_len ,N, heads, n_out, dropout=.1, seq_output=False):
        super().__init__()
        self.seq_output = seq_output
        self.encoder = Encoder(src_vocab, d_model, max_len ,N, heads, False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_out)
    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        
        if self.seq_output is False:
            e_outputs = e_outputs.mean(axis=1)
        
        e_outputs = self.dropout(e_outputs)
        output = self.out(e_outputs)
        return output
    
class HierarchicalTransformer(nn.Module):
    def __init__(self, char_vocab, word_vocab, d_model, max_len ,N, heads, n_out, outer_attention=True, dropout=.1, seq_output=False):
        super().__init__()
        self.seq_output = seq_output
        self.d_model = d_model
        self.word_embed = Embedder(word_vocab, d_model)
        self.char_encoder = TimeDistributed(Encoder(char_vocab, d_model, max_len, N, heads, outer_attention))
        self.word_encoder = EncoderWithoutEmbedding(d_model, max_len, N, heads)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_out)
        
    def forward(self, src, word_src, char_src_mask, word_src_mask):
        char_outputs = self.char_encoder(src, char_src_mask)
        char_outputs = char_outputs.view(src.shape[0],src.shape[1],src.shape[2],self.d_model)
        char_outputs = char_outputs.mean(axis=2)
        
        if word_src is not None:
            char_outputs = char_outputs + self.word_embed(word_src)

        e_outputs = self.word_encoder(char_outputs, word_src_mask)

        if self.seq_output is False:
            e_outputs = e_outputs.mean(axis=1)
        
        e_outputs = self.dropout(e_outputs)
        output = self.out(e_outputs)
        return output
    
class HAN(nn.Module):
    def __init__(self, char_vocab, word_vocab, d_model, N, n_out, heads=8, dropout=.1, seq_output=False):
        super().__init__()
        self.seq_output = seq_output
        self.d_model = d_model
        self.char_embed = Embedder(char_vocab, d_model)
        self.word_embed = Embedder(word_vocab, d_model)
        self.lstm1 = nn.LSTM(d_model, d_model, N, dropout=dropout, bidirectional=True,batch_first=True)
        self.lstm2 = nn.LSTM(d_model, d_model, N, dropout=dropout, bidirectional=True,batch_first=True)

        self.out1 = nn.Linear(d_model*2, d_model, bias=False)
        self.out2 = nn.Linear(d_model*2, d_model, bias=False)
        self.attention1 = AttentionWithContext(d_model)
        
        if seq_output == False:
            self.attention2 = AttentionWithContext(d_model)
        else:
            self.attention2 = MultiHeadAttention(heads, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_out, bias=False)
        
    def forward(self, src, word_src):
        char_outputs = self.char_embed(src)
        #h_n = nn.init.zeros_(torch.Tensor(2, src.shape[0], self.d_model))
        
        char_outputs = char_outputs.permute(1,0,2,3)
        
        char_outputs_l = []
        
        for i, char_output in enumerate(char_outputs):
            char_output, _ = self.lstm1(char_output)
            char_output = self.out1(char_output)
            _, char_output = self.attention1(char_output)
            
            char_outputs_l.append(char_output.unsqueeze(0))
        
        char_outputs = torch.cat(char_outputs_l, 0)
        
        char_outputs = char_outputs.permute(1,0,2)
        
        if word_src is not None:
            char_outputs = char_outputs + self.word_embed(word_src)
        
        e_outputs, _ = self.lstm2(char_outputs)
        e_outputs = self.out2(e_outputs)
        
        if self.seq_output == False:
            _, e_outputs = self.attention2(e_outputs)
        else:
            e_outputs = self.attention2(e_outputs,e_outputs,e_outputs)
        
        e_outputs = self.dropout(e_outputs)
        output = self.out(e_outputs)
        return output
    
class CMSA(nn.Module):
    def __init__(self, char_vocab, d_model, kernel_size, N, n_out, dropout=.1):
        super().__init__()

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.char_embed = Embedder(char_vocab, d_model)
        self.conv = nn.Conv1d(d_model,d_model,kernel_size=kernel_size,stride=1)
        
        self.lstm = nn.LSTM(d_model, d_model, N, dropout=dropout, bidirectional=True,batch_first=True)
        
        self.attention = AttentionWithContext(d_model*2)

        self.dropout = nn.Dropout(dropout)
        self.out1 = nn.Linear(d_model*4, d_model*2, bias=False)
        self.out2 = nn.Linear(d_model*2, d_model, bias=False)
        self.out3 = nn.Linear(d_model, d_model//2, bias=False)
        self.out4 = nn.Linear(d_model//2, n_out, bias=False)
        
    def forward(self, src):
        char_outputs = self.char_embed(src)
        char_outputs = F.pad(char_outputs, (0,0,0,self.kernel_size-1))
        print (char_outputs.shape)
        char_outputs = self.conv(char_outputs.permute(0,2,1)).permute(0,2,1)
        
        lstm_out, _ = self.lstm(char_outputs)
        _, attention = self.attention(lstm_out)
        
        e_outputs = torch.cat([lstm_out.mean(1),attention],-1)
        
        e_outputs = self.dropout(e_outputs)
        e_outputs = self.out1(e_outputs)
        e_outputs = self.dropout(e_outputs)
        e_outputs = self.out2(e_outputs)
        e_outputs = self.dropout(e_outputs)
        e_outputs = self.out3(e_outputs)
        e_outputs = self.dropout(e_outputs)
        
        output = self.out4(e_outputs)
        
        return output
    
class CS_ELMO(nn.Module):
    def __init__(self, char_vocab, word_vocab, d_model, max_len, N, n_out, dropout=.1, seq_output=False):
        super().__init__()
        
        self.seq_output = seq_output
        self.d_model = d_model

        self.char_embed = Embedder(char_vocab, d_model)
        self.word_embed = Embedder(word_vocab, d_model)
        
        self.bigram_conv = nn.Conv1d(d_model,d_model,kernel_size=2,stride=1)
        self.trigram_conv = nn.Conv1d(d_model,d_model,kernel_size=3,stride=1)
        
        self.bigram_pe = PositionalEncoder(d_model, max_len)
        self.trigram_pe = PositionalEncoder(d_model, max_len)
        
        self.lstm = nn.LSTM(d_model*2, d_model*2, N, dropout=dropout, bidirectional=True,batch_first=True)
        
        self.attention = AttentionWithContext(d_model*2)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model*2, d_model, bias=False)
        self.out = nn.Linear(d_model*4, n_out, bias=False)
        
    def forward(self, src, word_src):
        char_outputs = self.char_embed(src)
        
        char_outputs = char_outputs.permute(1,0,2,3)
        
        word_emb_l = []
        
        for i, char_output in enumerate(char_outputs):
            
            bigram_char_output = F.pad(char_output, (0,0,0,1))
            bigram_char_output = self.bigram_conv(bigram_char_output.permute(0,2,1)).permute(0,2,1)
            bigram_pe = self.bigram_pe(bigram_char_output)
            
            bigram_char_output += bigram_pe
            
            trigram_char_output = F.pad(char_output, (0,0,0,2))
            trigram_char_output = self.trigram_conv(trigram_char_output.permute(0,2,1)).permute(0,2,1)
            trigram_pe = self.trigram_pe(trigram_char_output)
            
            trigram_char_output += trigram_pe
            
            _, word_emb = self.attention(torch.cat([bigram_char_output, trigram_char_output], -1))
        
            word_emb_l.append(word_emb.unsqueeze(0))
        
        word_embs = torch.cat(word_emb_l,0)
        word_embs = word_embs.permute(1,0,2)
        
        if word_src is not None:
            word_embs = torch.cat([self.linear1(word_embs),self.word_embed(word_src)],-1)
            
        lstm_out, _ = self.lstm(word_embs)
        
        if self.seq_output is False:
            lstm_out = lstm_out.mean(axis=1)
        
        lstm_out = self.dropout(lstm_out)
        output = self.out(lstm_out)
        
        return output