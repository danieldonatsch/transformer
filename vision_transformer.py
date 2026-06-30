"""A "full" Vision transformer, including the embedding, class prediction token, all transformer layers and the output.

Code stems originally from:
https://colab.research.google.com/drive/19zAnmFvBU-vx64yriADswlZHbaU2NN2s?usp=sharing#scrollTo=raej4gAFWG0O
"""
import torch
import torch.nn as nn

from patch_embedding import PatchEmbedding
from vit_block import ViTBlock


class VisionTransformer(nn.Module):
    def __init__(self, num_img_chan: int, img_size: int, embedding_dim: int, patch_size: int,
                 num_transformer_blocks: int = 1, num_attention_heads: int = 1, mlp_factor: int = 1,
                 num_out_features: int = 10):
        """Initializes a vision transformer

        :param num_img_chan: (int) Number of image channels (1 for greyscale, 3 for RGN)
        :param embedding_dim: (int) dimension/length of the embedding vectors created by one image patch
        :param patch_size: (int) patch size (in one dimension), e.g. 4 if patches should be 4x4 pixels
        :param num_transformer_blocks: (int) Number of actual attention transformer blocks consisting of
                layer norm, attention, layer norm, mlp. (Default: 1)
        :param num_attention_heads: (int) Number of (parallel) attention heads (Default: 1)
        :param mlp_factor: (int) Factor by which the MLP should increase (and again decrease) (Default: 1).
        :param num_out_features: (int) Number of output features (Default: 10, since we use MNIST to test it).
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(num_img_chan, embedding_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embedding_dim))
        self.transformer_layers = nn.Sequential(*[ViTBlock(embedding_dim, num_attention_heads, mlp_factor)
                                                  for _ in range(num_transformer_blocks)])

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lin_out_layer = nn.Linear(in_features=embedding_dim, out_features=num_out_features, bias=True)

    def forward(self,x):
        # Convert input image into patch embeddings
        x = self.patch_embedding(x)
        # compute batch size
        batch_size = x.size(0)
        # generate the classification token and stack (concatenate) it
        cls_tokens = self.cls_token.expand(batch_size , -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add (sum) the positional embedding
        x = x + self.pos_embed
        # Do the magic, i.e. send through all transformers.
        x = self.transformer_layers(x)
        # Chop the classification token and send this through the final out layers.
        x = x[:, 0]
        x = self.layer_norm(x)
        return self.lin_out_layer(x)
