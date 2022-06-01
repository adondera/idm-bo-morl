import numpy as np

import gym


class DiscreteMountainCar3Distance(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 3
        self.reward_names = ["-distance_to_left_hill", "-distance_to_start", "-distance_to_right_hill"]

    def reward(self, reward: float) -> tuple[tuple[float, float, float], float]:
        """
        :param reward: The reward sampled from the environment (-1 or 0 if it reaches the goal)
        :return: A tuple containing the multi-objective reward (r) and the environment reward (z)
        The multi-objective reward is defined based on the negative of the distance to the left hill, the center, and
        the right hill
        """
        current_position, current_velocity = self.env.unwrapped.state
        goal_position = self.env.unwrapped.goal_position
        left_hill_position = self.env.unwrapped.min_position
        start_position = -0.5
        distance_to_right_hill = abs(current_position - goal_position)
        distance_to_left_hill = abs(current_position - left_hill_position)
        distance_to_start = abs(current_position - start_position)
        return (-distance_to_left_hill, -distance_to_start, -distance_to_right_hill), reward


class DiscreteMountainCarVelocity(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 1
        self.reward_names = ["current_velocity"]

    def reward(self, reward: float) -> tuple[tuple[float,], float]:
        """
        :param reward: The reward sampled from the environment (-1 or 0 if it reaches the goal)
        :return: A tuple containing the multi-objective reward (r) and the environment reward (z)
        The multi-objective reward contains only the velocity of the agent
        """
        current_position, current_velocity = self.env.unwrapped.state
        return (current_velocity,), reward


class DiscreteMountainCarNormal(gym.RewardWrapper):
    def __init__(self, env, max_episode_length=None):
        super().__init__(env)
        self.env = env
        self.numberPreferences = 1
        self.reward_names = ["global_reward"]
        self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
        if max_episode_length is not None:
            self.env._max_episode_steps = max_episode_length

    def rescale(self, state):
        return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)], dtype=np.single)

    def step(self, action):
        ns, r, d, x = self.env.step(action)
        return self.rescale(ns), self.reward(r), d, x

    def reset(self):
        return self.rescale(self.env.reset())

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed()

    def reward(self, reward: float) -> tuple[tuple[float,], float]:
        return (reward,), reward


# class RescaledEnv:
#     def __init__(self, env, max_episode_length=None):
#         self.env = env
#         self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
#         if max_episode_length is not None: self.env._max_episode_steps = max_episode_length
#
#     def rescale(self, state):
#         return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)])
#
#     def step(self, action):
#         ns, r, d, x = self.env.step(action)
#         return self.rescale(ns), r, d, x
#
#     def reset(self):
#         return self.rescale(self.env.reset())
#
#     def render(self, mode="human"):
#         self.env.render(mode)
#
#     def close(self):
#         self.env.close()
#
#     def seed(self, seed=None):
#         return self.env.seed()

# Use this code to play the environment or for debugging purposes

# env = DiscreteMountainCar3Distance(gym.make("MountainCar-v0"))
# env.reset()
# print(env.step(0))

# play(env)
