"""This is the actual transformer block from a vision transformer.

It's main difference is, that the layer normalization is beforehand, while the "text transformer" does it afterwards.
At least according to the graphics in the original papers.

Code stems originally from:
https://colab.research.google.com/drive/19zAnmFvBU-vx64yriADswlZHbaU2NN2s?usp=sharing#scrollTo=raej4gAFWG0O
"""
import torch.nn as nn

from multilayer_perceptron import MultilayerPerceptron


class ViTBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 1, mlp_factor: int = 1):
        """Initialization of a Vision Transformer Block

        :param embedding_dim: (int) Dimension of the embeddings
        :param num_heads: (int) Number of (parallel) heads (Default: 1)
        :param mlp_factor: (int) Factor by which the MLP should increase (and again decrease).
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.multi_layer_perceptron = MultilayerPerceptron(embedding_dim, mlp_factor)

    def forward(self, x):
        # First part: Normalization, attention and then summing to the input / residual
        normalized_x = self.layer_norm_1(x)
        attention_output = self.self_attention(normalized_x, normalized_x, normalized_x)[0]
        # Add residual to the attention output
        x = x + attention_output
        # Second part: again normalization, MLP and the skip/residual
        normalized_x = self.layer_norm_2(x)
        mlp_output = self.multi_layer_perceptron(normalized_x)
        x = x + mlp_output
        return x
