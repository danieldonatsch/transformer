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

import torch
import lightning as L ## Lightning makes it easier to write, optimize and scale our code

from decoder_only_transformer import DecoderOnlyTransformer
from mini_language_model import MiniLanguageModel
from data_loader import id_to_token, token_to_id, get_dataloader



class Experiment:

    def __init__(self, model=None):

        self.model = model


        if self.model is None:
            self.model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)

    def run_model(self, debug_mode=False,
                  test_phrase="llms can generate , summarize , translate and parse text in many contexts"):
        print("run_model() called")
        self.model.debug_mode = debug_mode

        test_phrase_tokens = test_phrase.split(' ')
        n = min(self.model.max_len, len(test_phrase_tokens)-1)
        ## Now create the input for the transformer...
        model_input = torch.tensor([token_to_id[t] for t in test_phrase_tokens[:n]])

        print("Test phrase:", test_phrase)
        print("Input:", " ".join([t for t in test_phrase_tokens[:n]]))

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
        max_length = len(test_phrase_tokens) + 1
        print(max_length, input_length)
        for i in range(input_length, max_length):
            if (predicted_id == token_to_id["."]):  # if the prediction is <EOS>, then we are done
                break

            model_input = torch.cat((model_input, predicted_id))
            if model_input.size(0) > self.model.max_len:
                model_input = model_input[1:]

            predictions = self.model(model_input)
            predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
            predicted_ids = torch.cat((predicted_ids, predicted_id))

        ## Now printout the predicted output phrase.
        print("Predicted Tokens:", " ".join([id_to_token[id.item()] for id in predicted_ids]))
        #for id in predicted_ids:
        #    print("\t", id_to_token[id.item()])

    def train_model(self, debug_mode=False):

        self.model.debug_mode = debug_mode
        dataloader = get_dataloader(n_token=self.model.max_len, debug_mode=debug_mode)

        trainer = L.Trainer(max_epochs=30)
        trainer.fit(self.model, train_dataloaders=dataloader)






if __name__ == '__main__':
    script_start_time = time.time()

    class UserArgs():
        debug_mode = True
        do_training = False
        # The maximum number of (input) tokens
        max_num_tokens = 6

    args = UserArgs()

    ## We set the seed so that we get each time the same result.
    L.seed_everything(seed=42)

    experiment = Experiment(
        #model=DecoderOnlyTransformer(num_tokens=len(token_to_id), d_embedding=128,
        #                             d_key=64, max_len=args.max_num_tokens)
        model = MiniLanguageModel(num_tokens=len(token_to_id), d_embedding=128,
                                  d_key_query_space=64, max_len=args.max_num_tokens)
    )
    print("Run untrained model")
    experiment.run_model(debug_mode=args.debug_mode)
    if args.do_training:
        print("Train model")
        experiment.train_model(debug_mode=args.debug_mode)
        print("Run trained model")

        test_phrase = "llms can generate , summarize , translate and parse text in many contexts , and are a foundational technology behind modern chatbots"
        experiment.run_model(debug_mode=args.debug_mode, test_phrase=test_phrase)

    print("Script finished after", time.time() - script_start_time, "seconds")

