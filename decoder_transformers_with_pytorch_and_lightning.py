"""
Original Source:
https://github.com/StatQuest/decoder_transformer_from_scratch/blob/main/decoder_transformers_with_pytorch_and_lightning_v2.ipynb
"""
import time

## First, check to see if lightning is installed, if not, install it.
import pip
try:
  __import__("lightning")
except ImportError:
  pip.main(['install', "lightning"])

import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # This gives us the softmax() and argmax()
from torch.optim import Adam ## We will use the Adam optimizer, which is, essentially,
                             ## a slightly less stochastic version of stochastic gradient descent.
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders

import lightning as L ## Lightning makes it easier to write, optimize and scale our code

from attention import Attention
from decoder_only_transformer import DecoderOnlyTransformer
from position_encoding import PositionEncoding
from my_data import id_to_token, token_to_id, get_dataloader



class Experiment:

    def __init__(self, model=None):

        self.model = model


        if self.model is None:
            self.model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)

    def run_model(self):
        print("run_model() called")

        ## Now create the input for the transformer...
        model_input = torch.tensor([token_to_id["what"],
                                    token_to_id["is"],
                                    token_to_id["statquest"],
                                    token_to_id["<EOS>"]])
        input_length = model_input.size(dim=0)

        ## Now get get predictions from the model
        predictions = self.model(model_input)
        ## NOTE: "predictions" is the output from the fully connected layer,
        ##      not a softmax() function. We could, if we wanted to,
        ##      Run "predictions" through a softmax() function, but
        ##      since we're going to select the item with the largest value
        ##      we can just use argmax instead...
        ## ALSO NOTE: "predictions" is a matrix, with one row of predicted values
        ##      per input token. Since we only want the prediction from the
        ##      last row (the most recent prediction) we use reverse index for the
        ##      row, -1.
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        ## We'll store predicted_id in an array, predicted_ids, that
        ## we'll add to each time we predict a new output token.
        predicted_ids = predicted_id

        ## Now use a loop to predict output tokens until we get an
        ## <EOS> token.
        max_length = 6
        for i in range(input_length, max_length):
            if (predicted_id == token_to_id["<EOS>"]):  # if the prediction is <EOS>, then we are done
                break

            model_input = torch.cat((model_input, predicted_id))

            predictions = self.model(model_input)
            predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
            predicted_ids = torch.cat((predicted_ids, predicted_id))

        ## Now printout the predicted output phrase.
        print("Predicted Tokens:\n")
        for id in predicted_ids:
            print("\t", id_to_token[id.item()])

    def train_model(self):

        dataloader = get_dataloader()

        trainer = L.Trainer(max_epochs=30)
        trainer.fit(self.model, train_dataloaders=dataloader)






if __name__ == '__main__':
    script_start_time = time.time()

    experiment = Experiment()
    print("Run untrained model")
    experiment.run_model()
    print("Train model")
    experiment.train_model()
    print("Run trained model")
    experiment.run_model()

    print("Script finished after", time.time() - script_start_time, "seconds")

