import torch.nn as nn
import torch, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term) 

        self.register_buffer('pe', pe.unsqueeze(0))

    
    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, image_size, patch_size, n_channels):
        super(PatchEmbedding, self).__init__()

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # B: Batch Size
        # C: Image Channels
        # H: Image Height
        # W: Image Width
        # P_col: Patch Column
        # P_row: Patch Row
        self.linear_projection = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size) # returns B, d_model, p_col, p_row
        
    def forward(self, x):
        x = self.linear_projection(x)
        x = x.flatten(2) # returns B, d_model, P (P is p_col*p_row)
        x = x.transpose(1, 2) # returns, B, P, d_model (accepted by transfprmer)

        return x
