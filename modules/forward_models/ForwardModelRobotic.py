import torch
from torch import nn
from torch.nn import *


class ForwardModelRobotic(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(ForwardModelRobotic, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.layers_encoder = [
            Linear(in_features=input_shape, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=action_dim, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        nn.init.xavier_uniform_(self.layers_encoder[2].weight)
        nn.init.xavier_uniform_(self.layers_encoder[4].weight)
        nn.init.xavier_uniform_(self.layers_encoder[6].weight)
        nn.init.xavier_uniform_(self.layers_encoder[8].weight)

        self.layers_model = [
            Linear(in_features=action_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=action_dim, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers_model[0].weight)
        nn.init.xavier_uniform_(self.layers_model[2].weight)
        nn.init.xavier_uniform_(self.layers_model[4].weight)
        nn.init.xavier_uniform_(self.layers_model[6].weight)
        nn.init.uniform_(self.layers_model[8].weight, -0.3, 0.3)

        self.encoder = Sequential(*self.layers_model)
        self.model = Sequential(*self.layers_model)

    def forward(self, state, action):
        f = self.encoder(state).detach()
        x = torch.cat([f, action], dim=1)
        x = self.model(x)
        predicted_state = x

        return predicted_state

    def encode(self, state):
        return self.encoder(state)

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)
            target = self.encoder(next_state)
            error = torch.mean(torch.pow(prediction - target, 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        loss = nn.functional.mse_loss(self(state, action), self.encoder(next_state).detach()) + self.variation_prior(state) + self.stability_prior(state, next_state)
        return loss
