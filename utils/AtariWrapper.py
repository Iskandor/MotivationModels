import gym
import numpy
from PIL import Image


class NopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsEnv, self).__init__(env)
        self.max_count = max_count

    def reset(self):
        self.env.reset()

        noops = numpy.random.randint(1, self.max_count + 1)

        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)

        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._obs_buffer = numpy.zeros((2,) + env.observation_space.shape, dtype=numpy.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height=96, width=96, frame_stacking=4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking - 1)):
            self.state[i + 1] = self.state[i].copy()
        self.state[0] = (numpy.array(img).astype(self.dtype) / 255.0).copy()

        return self.state


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)

        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
            reward = -1.0
        if lives == 0 and self.inital_lives > 0:
            reward = -1.0

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.inital_lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.raw_score_per_episode = 0.0
        self.raw_score_per_iteration = 0.0
        self.raw_reward = 0.0
        self.raw_reward_episode_sum = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.raw_reward_episode_sum += reward

        self.raw_reward = reward

        k = 0.01
        self.raw_score_per_iteration = (1.0 - k) * self.raw_score_per_iteration + k * reward

        if done:
            k = 0.05
            self.raw_score_per_episode = (1.0 - k) * self.raw_score_per_episode + k * self.raw_reward_episode_sum
            self.raw_reward_episode_sum = 0.0

        reward = numpy.clip(reward, -1.0, 1.0)
        return obs, reward, done, info


def AtariWrapper(env, height=96, width=96, frame_stacking=4, frame_skipping=4):
    env = NopOpsEnv(env)
    env = FireResetEnv(env)
    # env = SkipEnv(env, frame_skipping)
    env = MaxAndSkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)

    env.observation_space.shape = (frame_stacking, height, width)

    return env