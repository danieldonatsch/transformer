import torch
from torch.utils.data import TensorDataset, DataLoader

## first, we create a dictionary that maps vocabulary tokens to id numbers...
token_to_id = {'what': 0,
               'is': 1,
               'statquest': 2,
               'awesome': 3,
               '<EOS>': 4,  ## <EOS> = end of sequence
               }

## ...then we create a dictionary that maps the ids to tokens. This will help us interpret the output.
## We use the "map()" function to apply the "reversed()" function to each tuple (i.e. ('what', 0)) stored
## in the token_to_id dictionary. We then use dict() to make a new dictionary from the
## reversed tuples.
id_to_token = dict(map(reversed, token_to_id.items()))



def get_dataloader():
    ## NOTE: Because we are using a Decoder-Only Transformer, the inputs contain
    ##       the questions ("what is statquest?" and "statquest is what?") followed
    ##       by an <EOS> token followed by the response, "awesome".
    ##       This is because all of those tokens will be used as inputs to the Decoder-Only
    ##       Transformer during Training. (See the illustration above for more details)
    ## ALSO NOTE: When we train this way, it's called "teacher forcing".
    ##       Teacher forcing helps us train the neural network faster.
    inputs = torch.tensor([[token_to_id["what"],  ## input #1: what is statquest <EOS> awesome
                            token_to_id["is"],
                            token_to_id["statquest"],
                            token_to_id["<EOS>"],
                            token_to_id["awesome"]],

                           [token_to_id["statquest"],  # input #2: statquest is what <EOS> awesome
                            token_to_id["is"],
                            token_to_id["what"],
                            token_to_id["<EOS>"],
                            token_to_id["awesome"]]])

    ## NOTE: Because we are using a Decoder-Only Transformer the outputs, or
    ##       the predictions, are the input questions (minus the first word) followed by
    ##       <EOS> awesome <EOS>.  The first <EOS> means we're done processing the input question
    ##       and the second <EOS> means we are done generating the output.
    ##       See the illustration above for more details.
    labels = torch.tensor([[token_to_id["is"],
                            token_to_id["statquest"],
                            token_to_id["<EOS>"],
                            token_to_id["awesome"],
                            token_to_id["<EOS>"]],

                           [token_to_id["is"],
                            token_to_id["what"],
                            token_to_id["<EOS>"],
                            token_to_id["awesome"],
                            token_to_id["<EOS>"]]])

    ## Now let's package everything up into a DataLoader...
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset)



