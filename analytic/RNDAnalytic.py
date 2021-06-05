import numpy as np
from scipy.cluster.vq import kmeans, whiten


class RNDAnalytic:
    def __init__(self, config):
        self.states = []
        self.actions = []
        self.config = config

    def collect(self, state, action):
        self.states.append(state.squeeze(0).numpy())
        self.actions.append(action.squeeze(0).numpy())

    def generate_grid(self, s=100, a=25):
        self.states = np.stack(self.states)
        self.actions = np.stack(self.actions)

        state_prototypes, _ = kmeans(whiten(self.states), s)
        action_prototypes, _ = kmeans(whiten(self.actions), a)

        grid = []

        for state in state_prototypes:
            for action in action_prototypes:
                grid.append(np.concatenate([state, action]))

        grid = np.stack(grid)
        np.save('grid_{0}_{1}'.format(self.config.name, self.config.model), grid)



