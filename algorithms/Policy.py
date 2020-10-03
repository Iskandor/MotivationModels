import torch


class Policy:
    def __init__(self, network, actor_lr, weight_decay=0):
        self._actor = network
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr, weight_decay=weight_decay)

    def train(self, state0, action, td_error):
        self._actor_optimizer.zero_grad()
        logits_v = self._actor(state0)
        log_prob_v = torch.softmax(logits_v, dim=state0.ndim - 1)
        log_prob_actions_v = td_error * log_prob_v.gather(0, action)
        loss_v = -log_prob_actions_v.mean()
        loss_v.backward()
        self._actor_optimizer.step()

    def activate(self, state):
        with torch.no_grad():
            actions = torch.softmax(self._actor(state), dim=state.ndim - 1)
        return actions
