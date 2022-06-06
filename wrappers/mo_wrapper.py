from abc import abstractmethod
import gym


class MOWrapper(gym.RewardWrapper):
    @abstractmethod
    def reward(self, reward) -> tuple[tuple[float], float]:
        raise NotImplementedError
