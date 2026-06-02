import torch
import torch.nn as nn
from torch.optim import Adam

from position_encoding import PositionEncoding
from transformer import Transformer


class MiniLanguageModel(nn.Module):

    def __init__(self, num_tokens=4, d_embedding=2, d_key_query_space=2, max_len=6,
                 device=torch.device('cpu')):
        """Initializes our MiniLanguageModel, so en MLM (instead of an LLM)

        :param num_tokens: (int) The total number of tokens that exist
        :param d_embedding: (int) The size/dimension of the embedding
        :param d_key_query_space: (int) The dimension of the key-query-space in the attention block
        :param max_len: (int) The maximum length (i.e. num tokens) we allow as input
        """
        super().__init__()

        self.debug_mode = False
        self.device = device
        self.max_len = max_len

        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_embedding).to(self.device)

        self.pe = PositionEncoding(d_embedding=d_embedding,
                                   max_len=max_len).to(self.device)

        self.transformer = Transformer(d_embedding=d_embedding,
                                       d_key_query_space=d_key_query_space,
                                       device=self.device)

        self.fc_layer = nn.Linear(in_features=d_embedding, out_features=num_tokens).to(self.device)

        self.to(self.device)

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        if self.debug_mode:
            print("word embeddings:", word_embeddings.size())
            print("position encoding:", position_encoded.size())

        transformed_values = self.transformer(position_encoded)
        if self.debug_mode:
            print("self_attention_values.size():", transformed_values.size())


        fc_layer_output = self.fc_layer(transformed_values)

        if self.debug_mode:
            print("fc_layer_output.size():", fc_layer_output.size())

        return fc_layer_output

