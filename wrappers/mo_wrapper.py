import gym
from abc import abstractmethod


class MOWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        R, z = self.reward(reward)
        return observation, (R, z), done, info

    @abstractmethod
    def reward(self, reward) -> tuple[tuple[float], float]:
        raise NotImplementedError


class RescaledReward(MOWrapper):
    def __init__(self, env: MOWrapper, scale: list[int] = None):
        super().__init__(env)
        self.env = env
        self.numberPreferences = self.env.numberPreferences
        self.reward_names = self.env.reward_names
        self.scale = scale
        if scale is None:
            self.scale = [1] * self.env.numberPreferences

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        R, z = self.reward(reward)
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info

    def reward(self, reward):
        return self.env.reward(reward)
