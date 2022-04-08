from utils.RunningAverage import RunningStats


class CNDAnalytic:
    def __init__(self, nenv):
        self.int_rewards = RunningStats((1,), 'cpu', n=nenv)
        self.int_rewards_max = []
        self.int_rewards_mean = []
        self.int_rewards_std = []

    def add_reward(self, reward):
        self.int_rewards.update(reward, reduction='none')

    def reset(self, i):
        self.int_rewards.reset(i)

    def evaluate(self, i):
        max, mean, std = self.int_rewards.max[i].item(), self.int_rewards.mean[i].item(), self.int_rewards.std[i].item()

        self.int_rewards_max.append(max)
        self.int_rewards_mean.append(mean)
        self.int_rewards_std.append(std)

        return max, mean, std

