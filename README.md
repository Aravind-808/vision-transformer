Implementation of a basic Vision Transformer

stuff like MHA, encoder and embedding were taken from my other transformer repo with only minor tweaks

98% accuracy after 15 epochs and the following parameters

```
d_model = 64
n_classes = 10
img_size = (32,32)
patch_size = (8,8)
n_channels = 1
n_heads = 4
n_layers = 3
batch_size = 128
epochs = 15
alpha = 0.001
```
