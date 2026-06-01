import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from multilayer_perceptron import MultilayerPerceptron


class Transformer(nn.Module):

    def __init__(self, d_embedding: int, d_key_query_space: int, device=torch.device('cpu')):
        super().__init__()
        self.device = device

        self.self_attention = Attention(d_embedding=d_embedding, d_key=d_key_query_space).to(self.device)
        # Add an MLP Layer
        self.multilayer_perceptron = MultilayerPerceptron(d_embedding=d_embedding).to(self.device)

    def forward(self, embeddings):

        mask = torch.tril(torch.ones((embeddings.size(dim=0), embeddings.size(dim=0)), device=self.device))
        mask = mask == 0

        # Compute Self Attention
        self_attention_values = self.self_attention(embeddings, embeddings, embeddings, mask)

        # Add self attention to the embeddings
        transformed_values = embeddings + self_attention_values

        transformed_values = self.multilayer_perceptron(transformed_values)

        return transformed_values
