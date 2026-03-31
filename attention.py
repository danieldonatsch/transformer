import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, d_model=2):
        ## d_model = the number of embedding values per token.
        ##           In the transformer I used in the StatQuest: Transformer Neural Networks Clearly Explained!!!
        ##           d_model=2, so that's what we'll use as a default for now.
        ##           However, in "Attention Is All You Need" d_model=512

        super().__init__()

        self.d_model = d_model

        ## Initialize the Weights (W) that we'll use to create the
        ## query (q), key (k) and value (v) numbers for each token
        ## NOTE: Most implementations that I looked at include the bias terms
        ##       but I didn't use them in my video (since they are not in the
        ##       original Attention is All You Need paper).
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        ## NOTE: In this simple example, we are not training on the data in "batches"
        ##       However, by defining variables for row_dim and col_dim, we could
        ##       allow for batches by setting row_dim to 1 and col_com to 2.
        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        ## Create the query, key and values using the encodings
        ## associated with each token (token encodings)
        ##
        ## NOTE: For Encoder-Decoder Attention, the encodings for q come from
        ##       the decoder and the encodings for k and v come from the output
        ##       from the encoder.
        ##       In all of the other types of attention, the encodings all
        ##       come from the same source.
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        ## Compute attention scores
        ## the equation is (q * k^T)/sqrt(d_model)
        ## NOTE: It seems most people use "reverse indexing" for the dimensions when transposing k
        ##       k.transpose(dim0, dim1) will transpose k by swapping dim0 and dim1
        ##       In standard matrix notation, we would want to swap rows (dim=0) with columns (dim=1)
        ##       If we have 3 dimensions, because of batching, and the batch was the first dimension
        ##       And thus dims are defined batch = 0, rows = 1, columns = 2
        ##       then dim0=-2 = 3 - 2 = 1. dim1=-1 = 3 - 1 = 2.
        ##       Alternatively, we could put the batches in dim 3, and thus, dim 0 would still be rows
        ##       and dim 1 would still be columns. I'm not sure why batches are put in dim 0...
        ##
        ##       Likewise, the q.size(-1) uses negative indexing to rever to the number of columns in the query
        ##       which tells us d_model. Alternatively, we could ust q.size(2) if we have batches in the first
        ##       dimension or q.size(1) if we have batches in the 3rd dimension.
        ##
        ##       Since there are a bunch of ways to index things, I think the best thing to do is use
        ##       variables "row_dim" and "col_dim" instead of numbers...
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        if mask is not None:
            ## Here we are masking out things we don't want to pay attention to,
            ## like tokens that come after the current token.
            ## We can also use masking to block out the <PAD> token,
            ## which is used when we have a batch of inputs sequences
            ## and they are not all the exact same length. Because the batch is passed
            ## in as a matrix, each input sequence has to have the same length, so we
            ## add <PAD> to the shorter sequences so that they are all as long ast the
            ## longest sequence.
            ##
            ## We replace <PAD>, or tokens that come after the current token
            ## with a very large negative number so that the SoftMax() function
            ## will give all masked elements an output value (or "probability") of 0.
            scaled_sims = scaled_sims.masked_fill(mask=mask,
                                                  value=-1e9)  # I've also seen -1e20 and -9e15 used in masking

        ## Apply softmax to determine what percent of each token's value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        ## Scale the values by their associated percentages and add them up.
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
