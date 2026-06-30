"""Computes image patch embeddings to be fed into a Vision Transformer

Code stems originally from:
https://colab.research.google.com/drive/19zAnmFvBU-vx64yriADswlZHbaU2NN2s?usp=sharing#scrollTo=raej4gAFWG0O
"""
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, num_img_chan: int, embedding_dim: int, patch_size: int):
        """Initialises a PatchEmbedding class, which converts an image into (patch-wise) embeddings

        :param num_img_chan: (int) Number of image channels (1 for greyscale, 3 for RGN)
        :param embedding_dim: (int) dimension/length of the embedding vectors created by one image patch
        :param patch_size: (int) patch size (in one dimension), e.g. 4 if patches should be 4x4 pixels
        """
        super().__init__()
        self.patch_embed = nn.Conv2d(num_img_chan, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        # The output is [batch_size, embedding_dim, img_height/patch_size, img_width/patch_size]
        # So we flatten the last two dimensions, that the tensor becomes [batch_size, embedding_dim, num_patches]
        x = x.flatten(2)
        # To be aligned with the text transformer, we want a tensor of shape [batch_size, num_patches, embedding_dim].
        # We achieve this by transposing the last two dimensions.
        x = x.transpose(1, 2)
        return x