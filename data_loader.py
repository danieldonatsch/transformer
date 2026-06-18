import os
import pickle
import torch

from torch.utils.data import TensorDataset, DataLoader


# Parameters  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
text_file = "Wikipedia_LLM_clean.txt"
token_file = "Wikipedia_LLM_tokens.pickle"
data_dir = "data"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Token to ID map and vice-versa
with open(os.path.join(data_dir, token_file), 'rb') as of:
    token_to_id = pickle.load(of)

    id_to_token = dict(map(reversed, token_to_id.items()))


def get_dataloader(n_token, min_n_tokens=-1, batch_size=1, debug_mode=False):
    inputs = []
    labels = []

    if min_n_tokens < 1 or min_n_tokens > n_token:
        min_n_tokens = n_token

    with open(os.path.join(data_dir, text_file), 'r') as of:
        while line := of.readline():
            tokens = line.strip().split(" ")
            tokens.append('.')
            n = len(tokens)
            #print(tokens)
            if n > n_token:
                for i in range(n - n_token):
                    inputs.append([token_to_id[t] for t in tokens[i:i+n_token]])
                    labels.append([token_to_id[t] for t in tokens[i+1:i+n_token+1]])
                    #print("Inputs:", [t for t in tokens[i:i+n_token]])
                    #print("Labels:", [t for t in tokens[i+1:i+n_token+1]])
            elif n > min_n_tokens:
                padded_tokens = ['.'] * (n_token - n + 1) + tokens
                inputs.append([token_to_id[t] for t in padded_tokens[:-1]])
                labels.append([token_to_id[t] for t in padded_tokens[1:]])
            elif debug_mode:
                print("Not enough tokens in\n", tokens)

    input_tensor = torch.tensor(inputs)
    label_tensor = torch.tensor(labels)

    print("Number of tokens:", len(token_to_id))
    print("Size training set:", input_tensor.size())

    #print("Input Dim:", input_tensor.size())
    #print("Labels Dim:", label_tensor.size())
    ## Now let's package everything up into a DataLoader...
    dataset = TensorDataset(input_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
