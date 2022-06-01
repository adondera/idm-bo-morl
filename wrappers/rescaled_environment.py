import gym
import numpy as np


class RescaledEnv(gym.Wrapper):
    def __init__(self, env, max_episode_length=None):
        super().__init__(env)
        self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
        if max_episode_length is not None:
            self.env._max_episode_steps = max_episode_length

    def rescale(self, state):
        return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)], dtype=np.single)

    def step(self, action):
        ns, r, d, x = self.env.step(action)
        return self.rescale(ns), self.reward(r), d, x

    def reward(self, reward):
        return reward

    def reset(self):
        return self.rescale(self.env.reset())

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)