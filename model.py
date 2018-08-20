import torch 
import torch.nn as nn 
import onmt.modules as module
import copy
from multihead import Resblock
from linearlayer import Linear

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class IdentLayer1(nn.Module):
    def __init__(self, head_count, model_dim, dropout):
        super(IdentLayer1, self).__init__()
        self.multihead = module.MultiHeadedAttention(head_count, model_dim, dropout)
        self.layernorm = nn.LayerNorm(model_dim)
        self.resblock = Resblock(model_dim)
    def forward(self, x):
        output, top_attn = self.multihead(x,x,x)
        output = self.layernorm(output+x)
        output = self.resblock(output)

        return output, top_attn
class IdentLayer2(nn.Module):
    def __init__(self, head_count, model_dim, dropout):
        super(IdentLayer2, self).__init__()
        self.mulithead1 = module.MultiHeadedAttention(head_count, model_dim, dropout)
        self.multihead2 = module.MultiHeadedAttention(head_count, model_dim, dropout)
        self.layernorm = nn.LayerNorm(model_dim)
        self.resblock = Resblock(model_dim)
    def forward(self, x, y):
        output, _ = self.mulithead1(y,y,y)
        y = self.layernorm(output+y)
        output, top_attn = self.multihead2(x, x, y)
        output = self.resblock(output)

        return output, top_attn

class KeyphraseEncoder(nn.Module):
    def __init__(self, head_count, model_dim, dropout,n):
        super(KeyphraseEncoder, self).__init__()
        self.encoder = clones(IdentLayer1(head_count, model_dim, dropout), n)
    
    def forward(self, x):
        for layer in self.encoder:
            x, _ = layer(x)
        return x



class RankNet(nn.Module):
    def __init__(self, head_count, model_dim, dropout, n):
        super(RankNet, self).__init__()
        self.head_cout = head_count
        self.model_dim = model_dim
        self.encoder = KeyphraseEncoder(self.head_cout, self.model_dim, dropout, n)
        self.ranknet = clones(IdentLayer2(self.head_cout, self.model_dim, dropout), n-1)
        self.attn1 =  module.MultiHeadedAttention(self.head_cout, self.model_dim, dropout)
        self.attn2 =  module.MultiHeadedAttention(self.head_cout, self.model_dim, dropout)
        self.layernorm = nn.LayerNorm(model_dim)
    def forward(self, x, y):
        encoder_output = self.encoder(x)
        for layer in self.ranknet:
            y, weight = layer(encoder_output,y)
        output, _ = self.attn1(y, y, y)
        output = self.layernorm(output+y)
        output, top_attn = self.attn2(x,x,output)
        
        return top_attn
