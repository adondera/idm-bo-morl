import numpy as np

from wrappers.mo_wrapper import MOWrapper


class RescaledEnv(MOWrapper):
    def __init__(self, env, scale=None, max_episode_length=None):
        super().__init__(env, scale)
        self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
        if max_episode_length is not None:
            self.env._max_episode_steps = max_episode_length

    def rescale(self, state):
        return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)], dtype=np.single)

    def step(self, action):
        ns, r, d, x = self.env.step(action)
        R, z = self.reward(r)
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return self.rescale(ns), (R, z), d, x

    def reset(self):
        return self.rescale(self.env.reset())

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
