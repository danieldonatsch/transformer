import torch
import torch.nn as nn


class PositionEncoding(nn.Module):

    def __init__(self, d_embedding=2, max_len=6):
        """

        :param d_embedding: (int) the number of values in an embedding (embedding dimension)
        :param max_len: (int) maximum number of tokens we allow as input.
        """
        super().__init__()

        # We create a lookup table, pe, of position encoding values and initialize all of them to 0.
        # To do this, we will make a matrix of 0s that has max_len rows and d_embedding columns.
        pe = torch.zeros(max_len, d_embedding)

        # Now we create a sequence of numbers for each position that a token can have in the input (or output).
        # For example, if the input tokens where "I'm happy today!", then "I'm" would get the first
        # position, 0, "happy" would get the second position, 1, and "today!" would get the third position, 2.
        # NOTE: Since we are going to be doing math with these position indices to create the
        # positional encoding for each one, we need them to be floats rather than ints.
        #
        # Lastly, .unsqueeze(1) converts the single list of numbers that torch.arange creates into a matrix with
        # one row for each index, and all the indices in a single column.
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)

        # Here is where we start doing the math to determine the y-axis coordinates on the
        # sine and cosine curves.
        #
        # The positional encoding equations used in "Attention is all you need" are...
        #
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_embedding))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_embedding))
        #
        # ...and we see, within the sin() and cos() functions, we divide "pos" by some number that depends
        # on the index (i) and total number of PE values we want per token (d_embedding).
        #
        # NOTE: When the index, i, is 0 then we are calculating the y-axis coordinates on the **first pair**
        #       of sine and cosine curves. When i=1, then we are calculating the y-axis coordinates on the
        #       **second pair** of sine and cosine curves. etc. etc.
        #
        ## Now, pretty much everyone calculates the term we use to divide "pos" by first, and they do it with
        ## code that looks like this...
        ##
        ## div_term = torch.exp(torch.arange(start=0, end=d_embedding, step=2).float() * -(math.log(10000.0) / d_embedding))
        ##
        ## Now, at least to me, it's not obvious that div_term = 1/(10000^(2i/d_embedding)) for a few reasons:
        ##
        ##    1) div_term wraps everything in a call to torch.exp()
        ##    2) It uses log()
        ##    2) The order of the terms is different
        ##
        ## The reason for these differences is, presumably, trying to prevent underflow (getting too close to 0).
        ## So, to show that div_term = 1/(10000^(2i/d_embedding))...
        ##
        ## 1) Swap out math.log() for torch.log() (doing this requires converting 10000.0 to a tensor, which is my
        ##    guess for why they used math.log() instead of torch.log())...
        ##
        ## torch.exp(torch.arange(start=0, end=d_embedding, step=2).float() * -(torch.log(torch.tensor(10000.0)) / d_embedding))
        ##
        ## 2) Rearrange the terms...
        ##
        ## torch.exp(-1 * (torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_embedding, step=2).float() / d_embedding))
        ##
        ## 3) Pull out the -1 with exp(-1 * x) = 1/exp(x)
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_embedding, step=2).float() / d_embedding)
        ##
        ## 4) Use exp(a * b) = exp(a)^b to pull out the 2i/d_embedding term...
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)))^(torch.arange(start=0, end=d_embedding, step=2).float() / d_model)
        ##
        ## 5) Use exp(log(x)) = x to get the original form of the denominator...
        ##
        ## 1/(torch.tensor(10000.0)^(torch.arange(start=0, end=d_embedding, step=2).float() / d_embedding))
        ##
        ## 6) Bam.
        ##
        ## So, that being said, I don't think underflow is actually that big an issue. In fact, some coder at Hugging Face
        ## also doesn't think so, and their code for positional encoding in DistilBERT (a streamlined version of BERT, which
        ## is a transformer model)
        ## calculates the values directly - using the form of the equation found in original Attention is all you need
        ## manuscript. See...
        ## https://github.com/huggingface/transformers/blob/455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53
        ## So I think we can simplify the code, but I'm also writing all these comments to show that it is equivalent to what
        ## you'll see in the wild...
        ##
        ## Now let's create an index for the embedding positions to simplify the code a little more...
        embedding_index = torch.arange(start=0, end=d_embedding, step=2).float()
        ## NOTE: Setting step=2 results in the same sequence numbers that we would get if we multiplied i by 2.
        ##       So we can save ourselves a little math by just setting step=2.

        ## And now, finally, let's create div_term...
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_embedding)

        # Now we calculate the actual positional encoding values. Remember 'pe' was initialized as a matrix of 0s
        # with max_len (max number of input tokens) rows and d_embedding (number of embedding values per token) columns.
        pe[:, 0::2] = torch.sin(position * div_term)  # every other column, starting with the 1st, has sin() values
        pe[:, 1::2] = torch.cos(position * div_term)  # every other column, starting with the 2nd, has cos() values

        # Now we "register 'pe'.
        self.register_buffer('pe', pe)
        # "register_buffer()" ensures that 'pe' will be moved to wherever the model gets moved to.
        # So if the model is moved to a GPU, then, even though we don't need to optimize 'pe', it will
        # also be moved to that GPU. This, in turn, means that accessing 'pe' will be relatively fast compared
        # to having a GPU have to get the data from a CPU.

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(-2), :]
