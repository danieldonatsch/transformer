import torch
import torch.nn as nn
from torch.optim import Adam

from position_encoding import PositionEncoding
from transformer import Transformer


class MiniLanguageModel(nn.Module):

    def __init__(self, num_tokens=4, d_embedding=2, d_key_query_space=2, max_len=6,
                 num_layers=2, num_attention_heads=2, device=torch.device('cpu')):
        """Initializes our MiniLanguageModel, so en MLM (instead of an LLM)

        :param num_tokens: (int) The total number of tokens that exist
        :param d_embedding: (int) The size/dimension of the embedding
        :param d_key_query_space: (int) The dimension of the key-query-space in the attention block
        :param max_len: (int) The maximum length (i.e. num tokens) we allow as input
        :param num_layers: (int) The number of layers. One Layer is an attention block and an MLP block (Default: 2)
        :param num_attention_heads: (int) The number of attention heads in the transformer (Default: 2)
        :param device: torch.device (Default: 'cpu')
        """
        super().__init__()

        self.debug_mode = False
        self.device = device
        self.max_len = max_len

        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_embedding).to(self.device)

        self.pe = PositionEncoding(d_embedding=d_embedding,
                                   max_len=max_len).to(self.device)

        # Create the transformers
        self.transformers = [
            Transformer(d_embedding=d_embedding, d_key_query_space=d_key_query_space,
                        num_attention_heads=num_attention_heads, device=self.device)
            for _ in range(num_layers)
        ]

        self.un_embedding = nn.Linear(in_features=d_embedding, out_features=num_tokens).to(self.device)

        self.to(self.device)

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        if self.debug_mode:
            print("word embeddings:", word_embeddings.size())
            print("position encoding:", position_encoded.size())

        transformed_values = position_encoded
        for transformer in self.transformers:
            transformed_values = transformer(transformed_values)

        if self.debug_mode:
            print("self_attention_values.size():", transformed_values.size())

        word_prob = self.un_embedding(transformed_values)

        if self.debug_mode:
            print("word_prob.size():", word_prob.size())

        return word_prob
