import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class DANSE_Learner(nn.Module):

    def __init__(self, input_size, output_size, n_hidden, n_layers,
                 n_hidden_dense=32, num_directions=1, batch_first=True, min_delta=1e-2, use_cuda=True):
        super().__init__()
        # Defining some parameters
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        # Predefined:
        ## Use only the forward direction
        self.num_directions = num_directions

        ## The input tensors must have shape (batch_size,...)
        self.batch_first = batch_first

        # Defining the recurrent layers
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=self.batch_first).to(self.device)

        # Fully connected layer to be used for mapping the output
        # self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)

        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        # Add a dropout layer with 20% probability
        # self.d1 = nn.Dropout(p=0.2)

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0

    def forward(self, x):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = x.shape[0]

        # Obtain the RNN output
        r_out, _ = self.rnn(x)

        # Reshaping the output appropriately
        r_out_all_steps = r_out.contiguous().view(batch_size, -1, self.num_directions * self.hidden_dim)

        # Passing the output to one fully connected layer
        y = F.relu(self.fc(r_out_all_steps))

        # Means and variances are computed for time instants t=2, ..., T+1 using the available sequence
        mu_2T_1 = self.fc_mean(y)  # A second linear projection to get the means

        vars_2T_1 = F.softplus(self.fc_vars(y))  # A second linear projection followed by an activation function to get variances

        # The mean and variances at the first time step need to be computed only based on the previous hidden state
        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(batch_size, 1, -1)
        var_1 = F.softplus(self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(batch_size, 1, -1))

        # To get the means and variances for the time instants t=1, ..., T, we take the previous result and concatenate
        # all but last value to the value found at t=1. Concatenation is done along the sequence dimension
        mu = torch.cat(
            (mu_1, mu_2T_1[:, :-1, :]),
            dim=1
        )

        vars = torch.cat(
            (var_1, vars_2T_1[:, :-1, :]),
            dim=1
        )

        return mu, vars