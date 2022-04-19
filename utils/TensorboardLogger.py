from torch.utils.tensorboard import SummaryWriter
import time

class LogBoard:
    def __init__(self, config):
        t = time.localtime()
        self.writer = SummaryWriter(
            f'runs/model_{config.model}-descriptor-{config.descriptor}-{str(t.tm_hour) + "_" + str(t.tm_min)}')  # tensorboard --logdir=runs
        self.trials = 0

    def update_board_dop(self, step, ext_rew, steps, mean_rew, score):
        self.trials += 1
        self.writer.add_scalar('step', step, self.trials)
        self.writer.add_scalar('External_reward', ext_rew, self.trials)
        self.writer.add_scalar('No_steps', steps, self.trials)
        self.writer.add_scalar('Mean_reward', mean_rew, self.trials)
        self.writer.add_scalar('Score', score, self.trials)
