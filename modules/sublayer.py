import torch 
import torch.nn as nn
import linearlayer

class Resblock(nn.Module):
    
    def __init__(self, model_dim, dropout =0.1):
        super(Resblock, self).__init__()
        self.model_dim = model_dim
        self.resblock = nn.Sequential(
            nn.LayerNorm(model_dim),
            linearlayer.Linear(model_dim, 2*model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            linearlayer.Linear(2*model_dim, model_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        output = self.resblock(x)
        return output+x
