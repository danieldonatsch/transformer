import datetime
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from mini_language_model import MiniLanguageModel
from data_loader import id_to_token, token_to_id, get_dataloader
from utilities import get_device


class Experiment:

    def __init__(self, model=None, device=torch.device("cpu"), debug_mode=False):

        self.model = model
        self.debug_mode = debug_mode
        self.device = device
        self.tensorboard = None
        self.loss_function = nn.CrossEntropyLoss()

        if self.debug_mode:
            print("Device:", self.device)

    def run_model(self, debug_mode=False,
                  test_phrase="llms can generate , summarize , translate and parse text in many contexts"):

        print(f"run_model() with input test_phrase='{test_phrase}' called")

        self.model.eval()
        self.model.debug_mode = debug_mode

        test_phrase_tokens = test_phrase.split(' ')
        n = min(self.model.max_len, len(test_phrase_tokens)-1)
        ## Now create the input for the transformer...
        model_input = torch.tensor([token_to_id[t] for t in test_phrase_tokens[:n]])
        model_input = model_input.to(self.device)

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
        print("  Model Input:", " ".join([t for t in test_phrase_tokens[:n]]))
        print("  Predicted Tokens:", " ".join([id_to_token[id.item()] for id in predicted_ids]))
        #for id in predicted_ids:
        #    print("\t", id_to_token[id.item()])

    def train_model(self, epochs=10, lr=0.1, save_path: str = None, debug_mode=False):

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.tensorboard = SummaryWriter(save_path)

        optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        train_loader = get_dataloader(n_token=self.model.max_len,
                                      min_n_tokens=min(6, self.model.max_len),
                                      batch_size=args.batch_size,
                                      debug_mode=self.debug_mode)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            print("Train with learning rate", scheduler.get_last_lr())
            self.train_epoch(train_loader, optimizer, epoch)
            t1 = time.time()
            print(f"Training of epoch {epoch:2d} took {t1-t0:.1f} seconds")
            cor, mx = self.test_model(save_path, epoch)
            t2 = time.time()
            print(f"Test of epoch {epoch:2d} took {t2-t1:.1f} seconds. We had {cor}/{mx} correct predictions")
            scheduler.step()

            if save_path:
                torch.save(self.model.state_dict(),
                           os.path.join(save_path, f"{self.model.__class__.__name__}_epoch={epoch:02d}.pt"))

    def train_epoch(self, train_loader, optimizer, epoch):

        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)

            # If we train batches, we train kind of two batches!
            # Each token series is already a batch!
            output = torch.flatten(output, start_dim=0, end_dim=1)
            target = torch.flatten(target, start_dim=0, end_dim=1)

            loss = self.loss_function(output.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('  Train Epoch: {:2d} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            if not (self.tensorboard is None):
                self.tensorboard.add_scalar('Training Loss', loss.item(), ((epoch-1) * len(train_loader)) + batch_idx)
            if args.dry_run:
                break

    def test_model(self, save_path: str = None, epoch=0):

        test_phrases = [
            "llms can generate , summarize , translate and parse text in many contexts .",
            "a large language model is a neural network trained on a vast amount of text .",
            "as machine learning algorithms process numbers rather than text , the text must be converted to numbers .",
            "the llm can be fine - tuned through reinforcement learning to better satisfy this reward model ."
        ]

        self.model.eval()
        correct_predictions = 0
        max_possible = 0

        tot_loss = 0
        n_loss_vals = 0
        for test_phrase in test_phrases:
            test_phrase_tokens = test_phrase.split(' ')
            n = min(self.model.max_len, len(test_phrase_tokens) - 1)
            # Now create the input for the transformer...
            model_input = torch.tensor([token_to_id[t] for t in test_phrase_tokens[:n]])
            model_input = model_input.to(self.device)
            exp_output = torch.tensor([token_to_id[t] for t in test_phrase_tokens[n:]])
            exp_output = exp_output.to(self.device)
            max_possible += len(exp_output)

            predicted_ids = torch.empty((0), dtype=torch.float32)

            # Now use a loop to predict output tokens until we get an  <EOS> token.
            for i in range(len(exp_output)):
                # Let the model make a prediction
                predictions = self.model(model_input)
                predicted_id = torch.tensor([torch.argmax(predictions[-1, :])]).to(self.device)
                predicted_ids = torch.cat((predicted_ids, predicted_id))

                # Compute the loss
                loss = self.loss_function(predictions[-1, :], exp_output[i])
                tot_loss += loss.item()
                n_loss_vals += 1

                # Check if we predicted correctly:
                if predicted_id == exp_output[i]:
                    correct_predictions += 1
                else:
                    break
                # Check if we're at the end of the line
                if (predicted_id == token_to_id["."]):
                    break

                # So far, all predictions where correct and we're not at the end of the sentence.
                # Do a new prediction
                model_input = torch.cat((model_input, predicted_id))
                if model_input.size(0) > self.model.max_len:
                    model_input = model_input[1:]


            # Save results
            if save_path:
                file_name = f"test_results_epoch={epoch:02d}.txt"
                with open(os.path.join(save_path, file_name), 'a+') as of:
                    of.write(f"Test phrase: '{test_phrase}'\n")
                    of.write(f"Input: '{' '.join([t for t in test_phrase_tokens[:n]])}'\n")
                    of.write(f"Predicted Tokens: '{' '.join([id_to_token[id.item()] for id in predicted_ids])}'\n")
                    of.write(f"Expected Tokens:  '{' '.join([id_to_token[id.item()] for id in exp_output])}'\n\n")
                    of.write(f"Correct predicted words {correct_predictions:}/{max_possible}\n\n")

        if not (self.tensorboard is None):
            self.tensorboard.add_scalar('Test Loss', tot_loss/n_loss_vals, (epoch - 1))
            self.tensorboard.add_scalar('Test Correct Preds', correct_predictions, (epoch - 1))

        return correct_predictions, max_possible



def main(args):

    arg_file = None
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        arg_file = open(os.path.join(args.save_path, 'parameters.txt'), 'w')

    print("Arguments:")
    for arg in dir(args):
        if arg.startswith('_'):
            continue
        line = f"- {arg}: {args.__getattribute__(arg)}"
        print(line)
        if arg_file:
            arg_file.write(line + "\n")

    if arg_file:
        arg_file.close()
        arg_file = None

    experiment = Experiment(
        # model=DecoderOnlyTransformer(num_tokens=len(token_to_id), d_embedding=128,
        #                             d_key=64, max_len=args.max_num_tokens)
        model=MiniLanguageModel(num_tokens=len(token_to_id), d_embedding=128,
                                d_key_query_space=64, max_len=args.max_num_tokens,
                                num_layers=args.num_layers,
                                num_attention_heads=args.attention_heads,
                                device=get_device(args)),
        device=get_device(args),
        debug_mode=args.debug_mode
    )
    if args.load_weights:
        experiment.model.load_state_dict(torch.load(args.load_weights))
        print("Model Weights loaded from", args.load_weights)
        print("\n*** Run model with loaded wights")
        experiment.run_model(debug_mode=args.debug_mode)

    if args.do_training:
        print("\n*** Train model")
        experiment.train_model(debug_mode=args.debug_mode, lr=args.learning_rate,
                               epochs=args.epochs, save_path=args.save_path)

        print("\n*** Run trained model")
        test_phrase = "llms can generate , summarize , translate and parse text in many contexts , and are a foundational technology behind modern chatbots"
        experiment.run_model(debug_mode=args.debug_mode, test_phrase=test_phrase)


if __name__ == '__main__':
    script_start_time = time.time()

    class UserArgs():
        # The maximum number of (input) tokens
        max_num_tokens = 12
        num_layers = 16
        attention_heads = 16
        load_weights = None #'model_weights/MiniLanguageModel_epoch=01.pt'
        debug_mode = False
        # Training parameters
        do_training = True
        epochs = 100
        batch_size = 100
        learning_rate = 1.0
        lr_gamma = 0.5
        lr_step = 20
        no_gpu = False
        log_interval = 1_000
        dry_run = False
        save_path = 'results'

    args = UserArgs()

    if args.save_path == 'results':
        args.save_path = os.path.join(args.save_path, f"layers={args.num_layers}_aheads={args.attention_heads}")

    if args.batch_size > 1:
        args.log_interval = int(args.log_interval / args.batch_size)

    main(args)

    print("Script finished after", datetime.timedelta(seconds=(time.time() - script_start_time)))

