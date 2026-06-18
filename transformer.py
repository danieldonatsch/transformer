import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from multilayer_perceptron import MultilayerPerceptron


class Transformer(nn.Module):

    def __init__(self, d_embedding: int, d_key_query_space: int, num_attention_heads: int, device=torch.device('cpu')):
        """Initializes one transformer, consisting of (multi-headed) attention and a multi-layer perceptron.

        :param d_embedding: (int) dimension/size of the word/token embedding vector
        :param d_key_query_space: (int) dimension/size of the key-query-space vectors
        :param num_attention_heads: (int) Number of attention heads
        :param device: (torch.device) (Default: device=torch.device('cpu'))
        """
        super().__init__()
        self.device = device

        self.layer_norm1 = nn.LayerNorm(d_embedding).to(self.device)
        self.layer_norm2 = nn.LayerNorm(d_embedding).to(self.device)

        # Create Attention Blocks
        self.attention_blocks = [
            Attention(d_embedding=d_embedding, d_key=d_key_query_space).to(self.device)
            for _ in range(num_attention_heads)
        ]
        # Create MLP blocks
        self.multilayer_perceptron = MultilayerPerceptron(d_embedding=d_embedding).to(self.device)

    def forward(self, embeddings):

        n_tokens = embeddings.size(dim=-2)
        mask = torch.tril(torch.ones((n_tokens, n_tokens), device=self.device))
        mask = mask == 0

        # Apply all attention blocks of the layer
        self_attention_values = torch.zeros(embeddings.size(), device=self.device)
        for attention_block in self.attention_blocks:
            self_attention_values += attention_block(embeddings, embeddings, embeddings, mask)
        # Update the embeddings with the self attention values
        embeddings = embeddings + (self_attention_values / len(self.attention_blocks))
        # Apply layer norm
        embeddings = self.layer_norm1(embeddings)
        # Apply the MLP
        embeddings = self.multilayer_perceptron(embeddings)
        # Apply layer norm
        embeddings = self.layer_norm2(embeddings)

        return embeddings
