import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
import numpy as np
import copy

class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns 
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p) 
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None. 
    # I modified the code of dropout in PyTorch. 
    # Reference - https://github.com/bloodwass/mixout/blob/master/example.py

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1,"
                             " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError("A target tensor size must match with a input tensor size {},"
                             " but got {}". format(input.size(), target.size()))
        ctx.p = p    
        ctx.training = training
        
        if ctx.p == 0 or not ctx.training:
            return input
        
        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)
        
        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None

def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)

class MixLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p)) 
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer, 
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p) 
    # Reference - https://github.com/bloodwass/mixout/blob/master/example.py
    
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.target = target
        self.p = p
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, 
                                      self.p, self.training), self.bias)

    def extra_repr(self):
        type = 'drop' if self.target is None else 'mix' 
        return '{}={}, in_features={}, out_features={}, bias={}'.format(type+"out", self.p,
            self.in_features, self.out_features, self.bias is not None)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],requires_grad=False)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class SelfOuterAttention(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # transpose to get dimensions bs * h * sl * d_model
        # calculate attention using function we will define next
        scores, output = outer_product_attention(q, k, v, mask, self.dropout)
        
        self.attention_scores = scores
        
        output = self.out(output)
    
        return output
    
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

def outer_product_attention(q, k, v, mask=None, dropout=None):
    
    scores = torch.einsum('bnd,bmd->bnmd', q,k)
    
    #print (scores.shape)
    if mask is not None:
        if len(mask.size()) == 2:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        if len(mask.size()) == 3:
            mask = mask.unsqueeze(-1)
        #print (mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=2)

    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.einsum('bnmd,bmd->bnd',scores,v)
    
    return scores, output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, outer_attention=True, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        if outer_attention is False:
            self.attn = MultiHeadAttention(heads, d_model, dropout)
        else:
            self.attn = SelfOuterAttention(d_model, dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, N, heads, outer_attention=True):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, outer_attention), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class EncoderWithoutEmbedding(nn.Module):
    def __init__(self, d_model, max_len, N, heads, outer_attention=True):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, max_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, outer_attention), N)
        self.norm = Norm(d_model)
    def forward(self, x, mask):
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, src, mask):

        if len(src.size()) <= 2:
            return self.module(src, mask)

        # Squash samples and timesteps into a single axis
        src_reshape = src.contiguous().view(-1, src.size(-1))  # (samples * timesteps, input_size)
        mask_reshape = mask.contiguous().view(-1, mask.size(-1))  # (samples * timesteps, input_size)
        
        y = self.module(src_reshape, mask_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(src.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, src.size(1), y.size(-1))  # (timesteps, samples, output_size)
        
        return y

class AttentionWithContext(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(AttentionWithContext, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)