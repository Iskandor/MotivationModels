import torch


class ForwardInverseModelMotivation:
    def __init__(self, forward_model, forward_model_lr, inverse_model, inverse_model_lr, eta=1, upsilon=0.5, variant='A', memory_buffer=None, sample_size=0, device='cpu'):
        self._forward_model = forward_model
        self._forward_model_optimizer = torch.optim.Adam(self._forward_model.parameters(), lr=forward_model_lr)
        self._inverse_model = inverse_model
        self._inverse_model_optimizer = torch.optim.Adam(self._inverse_model.parameters(), lr=inverse_model_lr)
        self._upsilon = upsilon

        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._variant = variant
        self._device = device

    def train(self, state0, action, state1):
        if self._memory is not None:
            if len(self._memory) > self._sample_size:
                sample = self._memory.sample(self._sample_size)

                states = torch.stack(sample.state).squeeze(1)
                next_states = torch.stack(sample.next_state).squeeze(1)
                actions = torch.stack(sample.action).squeeze(1)

                self._forward_model_optimizer.zero_grad()
                loss = self._forward_model.loss_function(states, actions, next_states)
                loss.backward()
                self._forward_model_optimizer.step()

                self._inverse_model_optimizer.zero_grad()
                loss = self._inverse_model.loss_function(states, actions, next_states)
                loss.backward()
                self._inverse_model_optimizer.step()
        else:
            self._forward_model_optimizer.zero_grad()
            loss = self._forward_model.loss_function(state0, action, state1)
            loss.backward()
            self._forward_model_optimizer.step()

            self._inverse_model_optimizer.zero_grad()
            loss = self._inverse_model.loss_function(state0, action, state1)
            loss.backward()
            self._inverse_model_optimizer.step()

    def error(self, state0, action, state1):
        return self._forward_model.error(state0, action, state1), self._inverse_model.error(state0, action, state1)

    def reward(self, state0, action, state1):
        fm_error, im_error = self.error(state0, action, state1)
        reward = 0
        if self._variant == 'A':
            reward = torch.tanh(fm_error) * self._upsilon + torch.tanh(im_error) * (1 - self._upsilon)

        return reward * self._eta
