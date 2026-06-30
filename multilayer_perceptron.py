import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):

    def __init__(self, d_embedding=2, factor=4):
        """Initialize an MLP-Layer

        he MLP layer takes an embedding e, applies a linear/fully connected layer,
        which blows up the embedding to the given scale. To this up-scaled vector,
        ReLU is applied, than a second linear layer down-sizes the vector back to
        the original embedding size. Finally, this vector is added to the input
        embedding. The sum of this two vectors is the result of the MLP block.

        :param d_embedding: (int) dimension of the embedding
        :param factor: (int) the (up-)scaling factor used by the MLP-blockß
        """
        super().__init__()

        self.d_embedding=d_embedding
        self.factor=factor

        self.W_up = nn.Linear(in_features=d_embedding, out_features=int(factor*d_embedding), bias=True)
        self.W_dwn = nn.Linear(in_features=int(factor*d_embedding), out_features=d_embedding, bias=True)

    def forward(self, embedding):
        x = self.W_up(embedding)
        x = F.relu(x)
        x = self.W_dwn(x)

        return x

