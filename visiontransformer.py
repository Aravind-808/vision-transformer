import torch.nn as nn
from encoding_embedding import PositionalEncoding, PatchEmbedding
from encoder import Encoder

class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, image_size, patch_size, n_channels, n_heads, n_layers):
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0 
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_classes = n_classes
        self.image_size = image_size
        self.patch_size  = patch_size
        self.n_heads = n_heads 
        self.n_channels = n_channels

        self.num_patches = (self.image_size[0] * self.image_size[1]) // (self.patch_size[0]* self.patch_size[1])
        self.max_seq_len = self.num_patches+1

        self.patch_embedding = PatchEmbedding(self.d_model, self.image_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        self.encoder = nn.Sequential(*[Encoder(self.d_model, self.n_heads) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim =1)
        )
    
    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0])

        return x