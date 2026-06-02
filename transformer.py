import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from multilayer_perceptron import MultilayerPerceptron


class Transformer(nn.Module):

    def __init__(self, d_embedding: int, d_key_query_space: int, device=torch.device('cpu'),
                 num_layers=2, num_attention_heads=3):
        super().__init__()
        self.device = device

        self.attention_blocks = []
        self.multilayer_perceptrons = []

        for layer in range(num_layers):
            # Create Attention Blocks
            self.attention_blocks.append(
                [Attention(d_embedding=d_embedding, d_key=d_key_query_space).to(self.device)
                 for _ in range(num_attention_heads)]
            )
            # Create MLP blocks
            self.multilayer_perceptrons.append(
                MultilayerPerceptron(d_embedding=d_embedding).to(self.device)
            )

    def forward(self, embeddings):

        mask = torch.tril(torch.ones((embeddings.size(dim=0), embeddings.size(dim=0)), device=self.device))
        mask = mask == 0

        for i in range(len(self.attention_blocks)):
            # Apply all attention blocks of the layer
            self_attention_values = torch.zeros(embeddings.size(), device=self.device)
            for attention_block in self.attention_blocks[i]:
                self_attention_values += attention_block(embeddings, embeddings, embeddings, mask)
            # Update the embeddings with the self attention values
            embeddings = embeddings + self_attention_values
            # Apply the MLP
            embeddings = self.multilayer_perceptrons[i](embeddings)

        return embeddings
