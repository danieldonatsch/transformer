"""
Original Source:
https://github.com/StatQuest/decoder_transformer_from_scratch/blob/main/decoder_transformers_with_pytorch_and_lightning_v2.ipynb
"""
import os
import time

## First, check to see if lightning is installed, if not, install it.
import pip
try:
  __import__("lightning")
except ImportError:
  pip.main(['install', "lightning"])

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L ## Lightning makes it easier to write, optimize and scale our code

from decoder_only_transformer import DecoderOnlyTransformer
from mini_language_model import MiniLanguageModel
from data_loader import id_to_token, token_to_id, get_dataloader


def get_device(args) -> torch.device:
    """Checks if user requested GPU or CPU training and if GPUs are available. Also deals with processors architecture.

    :param args: args from ArgumentParser
    :return: torch.device
    """
    if args.no_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("GPU-training not possible")
    return torch.device("cpu")


class Experiment:

    def __init__(self, model=None, device=torch.device("cpu"), debug_mode=False):

        self.model = model
        self.debug_mode = debug_mode
        self.device = device

        if self.debug_mode:
            print("Device:", self.device)

    def run_model(self, debug_mode=False,
                  test_phrase="llms can generate , summarize , translate and parse text in many contexts"):
        print("run_model() called")
        self.model.eval()
        self.model.debug_mode = debug_mode

        test_phrase_tokens = test_phrase.split(' ')
        n = min(self.model.max_len, len(test_phrase_tokens)-1)
        ## Now create the input for the transformer...
        model_input = torch.tensor([token_to_id[t] for t in test_phrase_tokens[:n]])
        model_input = model_input.to(self.device)

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
        for i in range(input_length, max_length):
            if (predicted_id == token_to_id["."]):  # if the prediction is <EOS>, then we are done
                break

            model_input = torch.cat((model_input, predicted_id.to(self.device)))
            if model_input.size(0) > self.model.max_len:
                model_input = model_input[1:]

            predictions = self.model(model_input)
            predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
            predicted_ids = torch.cat((predicted_ids, predicted_id))

        ## Now printout the predicted output phrase.
        print("Predicted Tokens:", " ".join([id_to_token[id.item()] for id in predicted_ids]))
        #for id in predicted_ids:
        #    print("\t", id_to_token[id.item()])

    def train_model(self, epochs=10, lr=0.1, save_path: str = None, debug_mode=False):

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        train_loader = get_dataloader(n_token=self.model.max_len, debug_mode=self.debug_mode)

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, optimizer, epoch)
            #self.run_model()
            #scheduler.step()
            if save_path:
                torch.save(self.model.state_dict(),
                           os.path.join(save_path, f"{self.model.__class__.__name__}_epoch={epoch:02d}.pt"))

    def train_epoch(self, train_loader, optimizer, epoch):

        loss_function = nn.CrossEntropyLoss()
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)

            loss = loss_function(output.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {:2d} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


def main(args):

    experiment = Experiment(
        # model=DecoderOnlyTransformer(num_tokens=len(token_to_id), d_embedding=128,
        #                             d_key=64, max_len=args.max_num_tokens)
        model=MiniLanguageModel(num_tokens=len(token_to_id), d_embedding=128,
                                d_key_query_space=64, max_len=args.max_num_tokens,
                                device=get_device(args)),
        device=get_device(args),
        debug_mode=args.debug_mode
    )
    if args.load_weights:
        experiment.model.load_state_dict(torch.load(args.load_weights))
        print("Model Weights loaded from", args.load_weights)

    print("Run untrained model")
    experiment.run_model(debug_mode=args.debug_mode)

    if args.do_training:
        print("Train model")
        experiment.train_model(debug_mode=args.debug_mode, epochs=args.epochs, save_path=args.save_path)
        print("Run trained model")

        test_phrase = "llms can generate , summarize , translate and parse text in many contexts , and are a foundational technology behind modern chatbots"
        experiment.run_model(debug_mode=args.debug_mode, test_phrase=test_phrase)


if __name__ == '__main__':
    script_start_time = time.time()

    class UserArgs():
        # The maximum number of (input) tokens
        max_num_tokens = 6
        load_weights = 'model_weights/MiniLanguageModel_epoch=01.pt'
        debug_mode = False
        # Training parameters
        do_training = True
        epochs = 1
        no_gpu = False
        log_interval = 1_000
        dry_run = False
        save_path = 'model_weights'

    args = UserArgs()
    main(args)

    print("Script finished after", time.time() - script_start_time, "seconds")

