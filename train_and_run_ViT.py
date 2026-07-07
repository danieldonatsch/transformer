"""Trains and runs the Vision Transformer that we implemented.

We use MNIST as experiment, since we have already a CNN implemented which does MNIST-classification and we compare:
https://github.com/danieldonatsch/MNIST_GAN
"""
import datetime
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from parameters import Parameters
from utilities import get_device
from vision_transformer import VisionTransformer


class Experiment:
    def __init__(self, training_params: Parameters, model=None):
        self.model = model
        self.debug_mode = training_params.debug_mode
        self.device = get_device(training_params)
        self.tensorboard = None
        self.parameters = training_params
        self.loss_function = nn.CrossEntropyLoss()
        self.log_file_path = None

        self.train_loader = None
        self.test_loader = None

        if self.debug_mode:
            print("Device:", self.device)

        if not self.model:
            self.model = VisionTransformer(num_img_chan=1, img_size=28, # MNIST image size
                                           embedding_dim=self.parameters.embedding_dim,
                                           patch_size=self.parameters.patch_size,
                                           num_transformer_blocks=self.parameters.num_layers,
                                           num_attention_heads=self.parameters.attention_heads,
                                           mlp_factor=self.parameters.mlp_factors,
                                           input_dropout_rate=self.parameters.drop_out_rate,
                                           num_out_features=10 # MNIST
                                          )

    def get_mnist_data(self):
        """Generates the data loaders for test and training data with MNIST images

        :return:
        """

        train_kwargs = {'batch_size': self.parameters.batch_size, 'shuffle': True}
        test_kwargs = {'batch_size': self.parameters.batch_size}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('../data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        self.print("Training data:", len(dataset_train), "samples split into", len(self.train_loader), "batches")
        self.print("Test data:", len(dataset_test), "samples split into", len(self.test_loader), "batches")

    def run_vit(self):
        pass

    def train_vit(self):
        """Trains the vision transformer

        :return:
        """
        if self.parameters.save_path:
            os.makedirs(self.parameters.save_path, exist_ok=True)
            self.tensorboard = SummaryWriter(self.parameters.save_path)
            self.parameters.save_parameters('log.txt')
            self.log_file_path = os.path.join(self.parameters.save_path, 'log.txt')

        self.model.train()
        self.model.to(self.device)

        optimizer = optim.Adadelta(self.model.parameters(), lr=self.parameters.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.parameters.lr_step,
                                              gamma=self.parameters.lr_gamma)

        for epoch in range(1, self.parameters.epochs + 1):

            t0 = time.time()
            self.print("\nTrain with learning rate", scheduler.get_last_lr())

            train_loss = self.train_one_epoch(optimizer, epoch)

            t1 = time.time()
            self.print(f"Training of epoch {epoch:2d}:\n- time in sec: {t1-t0:.1f}\n- Loss per sample: {train_loss}")

            test_loss, test_acc = self.test_vit(epoch)

            t2 = time.time()
            self.print(f"Test after epoch {epoch:2d}:", f"- time in sec: {t2-t1:.1f}",
                       f"- Loss per sample: {test_loss}", f"- Accuracy: {test_acc}", sep='\n')

            scheduler.step()

            if self.parameters.save_path:
                torch.save(self.model.state_dict(),
                           os.path.join(self.parameters.save_path,
                                        f"{self.model.__class__.__name__}_epoch={epoch:02d}.pt"))

    def test_vit(self, epoch: int) -> tuple:
        """Runs a test over the test-set, computes accuracy and loss, draws it on the tensorboard but also returns it.

        :param epoch: (int) Number of the epoch
        :return: (tuple) test loss per sample, accuracy over the hole test set
        """
        self.model.eval()
        total_loss = 0
        tot_correct = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()

                batch_size = data.size(0)
                total_loss += loss.item()
                tot_correct += correct
                num_samples += batch_size

                if not (self.tensorboard is None):
                    it_num = ((epoch - 1) * len(self.test_loader)) + batch_idx
                    self.tensorboard.add_scalar('Test Loss', loss.item() / batch_size, it_num)
                    self.tensorboard.add_scalar('Test Accuracy', correct / batch_size, it_num)

        rand_samples = [sample_no for sample_no in range(0, batch_size, batch_size//16)]
        plt.figure(figsize=(32, 18))
        for i, sample_ind in enumerate(rand_samples[:16]):
            idx = sample_ind
            plt.subplot(4, 4, i + 1)
            plt.title(f"Prediction: {pred[idx].item()}, Target {target[idx].item()}")
            plt.imshow(data[idx, 0, :, :].cpu(), cmap='gray', vmin=0.0, vmax=1.0) #vmin=-0.4242, vmax=2.8215)
        plt.savefig(os.path.join(self.parameters.save_path, f'test_samples_epoch={epoch:02d}.png'))

        return total_loss / num_samples, tot_correct / num_samples

    def train_one_epoch(self, optimizer, epoch):
        """Does a training epoch

        """

        total_loss, num_samples = 0, 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            optimizer.step()

            num_samples += batch_size
            total_loss += loss.item()

            if not (self.tensorboard is None):
                it_num = ((epoch - 1) * len(self.train_loader)) + batch_idx
                self.tensorboard.add_scalar('Training Loss', loss.item() / batch_size, it_num)

            if self.parameters.dry_run:
                break

        return total_loss / num_samples

    def print(self, *args, sep=' '):
        message = sep.join([str(arg) for arg in args])
        if self.log_file_path:
            with open(self.log_file_path, 'a') as of:
                of.write(message + "\n")
        print(message)


if __name__ == '__main__':
    script_start_time = time.time()

    params = Parameters(result_dir='res_vit')
    params.embedding_dim = 64
    params.patch_size = 7
    params.attention_heads = 8
    params.num_layers = 8
    params.mlp_factors = 2
    params.learning_rate = 0.01
    params.debug_mode = True
    params.lr_gamma = 0.1
    params.lr_step = 10
    params.drop_out_rate = 0.3

    exp = Experiment(training_params=params)

    exp.get_mnist_data()

    if params.do_training:
        exp.train_vit()

    print("Script finished after", datetime.timedelta(seconds=(time.time() - script_start_time)))




