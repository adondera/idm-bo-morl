from abc import abstractmethod
import gym


class MOWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=None):
        super().__init__(env)
        self.scale = scale
        if scale is None:
            self.scale = [1] * self.env.numberPreferences

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        R, z = self.reward(reward)
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info

    @abstractmethod
    def reward(self, reward) -> tuple[tuple[float], float]:
        raise NotImplementedError
