import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L

from attention import Attention
from position_encoding import PositionEncoding


class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()

        ## We are set the seed so that you can get the same results as me.
        L.seed_everything(seed=42)

        ## NOTE: In this simple example, we are just using a "single layer" decoder.
        ##       If we wanted to have multiple layers of decoder, then we would
        ##       take the output of one decoder module and use it as input to
        ##       the next module.

        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)

        self.pe = PositionEncoding(d_model=d_model,
                                   max_len=max_len)

        self.self_attention = Attention(d_model=d_model)
        ## NOTE: In this simple example, we are not doing multi-head attention
        ## If we wanted to do multi-head attention, we could
        ## initialize more Attention objects like this...
        ##
        ## self.self_attention_2 = Attention(d_model=d_model)
        ## self.self_attention_3 = Attention(d_model=d_model)
        ##
        ## If d_model=2, then using 3 self_attention objects would
        ## result in d_model*3 = 6 self-attention values per token,
        ## so we would need to initialize
        ## a fully connected layer to reduce the dimension of the
        ## self attention values back down to d_model like this:
        ##
        ## self.reduce_attention_dim = nn.Linear(in_features=(num_attention_heads*d_model), out_features=d_model)

        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        ## For the decoder-only transformer, we need to use "masked self-attention" so that
        ## when we are training we can't cheat and look ahead at
        ## what words come after the current word.
        ## To create the mask we are creating a matrix where the lower triangle
        ## is filled with 0, and everything above the diagonal is filled with 0s.
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=self.device))
        ## NOTE: The device=self.device is needed because we are creating a new
        ##       tensor, mask, in the forward() method, which, by default, goes
        ##       to the CPU. If all we have is a CPU, then we don't need it, but
        ##       if we want to train on a GPU, we need to make sure mask goes
        ##       there too. Using self.device allows us tyo not worry about whether
        ##       or not we are using a GPU or CPU or whatever, it will make sure
        ##       mask is where it needs to go.

        ## We then replace the 0s above the digaonal, which represent the words
        ## we want to be masked out, with "True", and replace the 1s in the lower
        ## triangle, which represent the words we want to include when we calcualte
        ## self-attention for a specific word in the output, with "False".
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        ## NOTE: If we were doing multi-head attention, we would
        ## calculate the self-attention values with the other attention objects
        ## like this...
        ##
        ## self_attention_values_2 = self.self_attention_2(...)
        ## self_attention_values 3 = self.self_attention_3(...)
        ##
        ## ...then we would concatenate all the self attention values...
        ##
        ## all_self_attention_values = torch.cat(self_attention_values_1, ...)
        ##
        ## ...and then run them through reduce_dim to get back to d_model values per token
        ##
        ## final_self_attention_values = self.reduce_attention_dim(all_self_attention_values)

        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output

    def configure_optimizers(self):
        ## configure_optimizers() simply passes the parameters we want to
        ## optimize to the optimzes and sets the learning rate
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        ## training_step() is called by Lightning trainer when
        ## we want to train the model.
        input_tokens, labels = batch  # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss
