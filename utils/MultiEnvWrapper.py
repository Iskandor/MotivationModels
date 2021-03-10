import threading
import multiprocessing
import time
import numpy
import gym


class MultiEnvSeq:
    def __init__(self, env_name, wrapper, envs_count):

        try:
            dummy_env = gym.make(env_name)
            if wrapper is not None:
                dummy_env = wrapper(dummy_env)
        except:
            dummy_env = wrapper(env_name)

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        self.envs = []

        for i in range(envs_count):

            try:
                env = gym.make(env_name)
                if wrapper is not None:
                    env = wrapper(env)
            except:
                env = wrapper(env_name)

            self.envs.append(env)

    def close(self):
        pass

    def reset(self, env_id):
        return self.envs[env_id].reset()

    def step(self, actions):
        obs = []
        reward = []
        done = []
        info = []

        for e in range(len(self.envs)):
            _obs, _reward, _done, _info = self.envs[e].step(actions[e])

            obs.append(_obs)
            reward.append(_reward)
            done.append(_done)
            info.append(_info)

        return obs, reward, done, info

    def render(self, env_id):
        self.envs[env_id].render()

    def get(self, env_id):
        return self.envs[env_id]


def env_process_main(id, inq, outq, env_name, wrapper, count):
    print("env_process_main = ", id, count, env_name)
    envs = []

    for _ in range(count):

        try:
            env = gym.make(env_name)
            if wrapper is not None:
                env = wrapper(env)
        except:
            env = wrapper(env_name)

        envs.append(env)

    while True:
        val = inq.get()

        if val[0] == "end":
            break

        elif val[0] == "reset":
            env_id = val[1]

            _obs = envs[env_id].reset()

            outq.put(_obs)

        elif val[0] == "step":
            actions = val[1]

            obs = []
            rewards = []
            dones = []
            infos = []

            for i in range(count):
                _obs, _reward, _done, _info = envs[i].step(actions[i])

                obs.append(_obs)
                rewards.append(_reward)
                dones.append(_done)
                infos.append(_info)

            outq.put((obs, rewards, dones, infos))

        elif val[0] == "render":
            env_id = val[1]
            envs[env_id].render()

        elif val[0] == "get":
            env_id = val[1]
            outq.put(envs[env_id])


class MultiEnvParallel:
    def __init__(self, env_name, wrapper, envs_count, envs_per_thread=8):
        try:
            dummy_env = gym.make(env_name)
            if wrapper is not None:
                dummy_env = wrapper(dummy_env)
        except:
            dummy_env = wrapper(env_name)

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        self.inq = []
        self.outq = []
        self.workers = []

        self.envs_count = envs_count
        self.threads_count = envs_count // envs_per_thread
        self.envs_per_thread = envs_per_thread

        print("MultiEnvParallel")
        print("envs_count      = ", self.envs_count)
        print("threads_count   = ", self.threads_count)
        print("envs_per_thread = ", self.envs_per_thread)
        print("\n\n")

        for i in range(self.threads_count):
            inq = multiprocessing.Queue()
            outq = multiprocessing.Queue()

            worker = multiprocessing.Process(target=env_process_main, args=(i, inq, outq, env_name, wrapper, envs_per_thread))

            self.inq.append(inq)
            self.outq.append(outq)
            self.workers.append(worker)

        for i in range(self.threads_count):
            self.workers[i].start()

    def close(self):
        for i in range(len(self.workers)):
            self.inq[i].put(["end"])

        for i in range(len(self.workers)):
            self.workers[i].join()

    def reset(self, env_id):
        thread, id = self._position(env_id)

        self.inq[thread].put(["reset", id])

        obs = self.outq[thread].get()
        return obs

    def render(self, env_id):
        thread, id = self._position(env_id)

        self.inq[thread].put(["render", id])

    def step(self, actions):
        for j in range(self.threads_count):
            _actions = []
            for i in range(self.envs_per_thread):
                _actions.append(actions[j * self.envs_per_thread + i])

            self.inq[j].put(["step", _actions])

        obs = []
        reward = []
        done = []
        info = []

        for j in range(self.threads_count):
            _obs, _reward, _done, _info = self.outq[j].get()

            for i in range(self.envs_per_thread):
                obs.append(_obs[i])
                reward.append(_reward[i])
                done.append(_done[i])
                info.append(_info[i])

        return obs, reward, done, info

    def get(self, env_id):
        thread, id = self._position(env_id)

        self.inq[thread].put(["get", id])

        return self.outq[thread].get()

    def _position(self, env_id):
        return env_id // self.envs_per_thread, env_id % self.envs_per_thread