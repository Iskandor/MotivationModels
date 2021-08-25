import numpy
import psutil
import torch
import umap
from etaprogress.progress import ProgressBar


class DOPAnalytic:
    def __init__(self):
        pass

    @staticmethod
    def head_analyze(env, agent):
        step_limit = 2000
        steps = 0

        bar = ProgressBar(step_limit, max_width=80)

        states = []
        actions = []
        head_indices = []

        while steps < step_limit:
            state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_steps = 0

            while not done:
                action0, head_index, _ = agent.get_action(state0)
                next_state, _, done, _ = env.step(action0.squeeze(0).numpy())
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                states.append(state0.flatten().numpy())
                actions.append(action0.squeeze(0).numpy())
                head_indices.append(head_index.item())
                train_steps += 1

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
                states = states[:step_limit]
                actions = actions[:step_limit]
                head_indices = head_indices[:step_limit]
            bar.numerator = steps
            print(bar)

        reducer = umap.UMAP(n_jobs=psutil.cpu_count(logical=True))

        states = reducer.fit_transform(numpy.stack(states))
        actions = reducer.fit_transform(numpy.stack(actions))
        head_indices = numpy.stack(head_indices)

        return states, actions, head_indices
