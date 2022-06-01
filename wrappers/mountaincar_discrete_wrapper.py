from wrappers.rescaled_environment import RescaledEnv


class DiscreteMountainCar3Distance(RescaledEnv):
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


class DiscreteMountainCarVelocity(RescaledEnv):
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


class DiscreteMountainCarNormal(RescaledEnv):
    def __init__(self, env, max_episode_length=None):
        super().__init__(env, max_episode_length)
        self.numberPreferences = 1
        self.reward_names = ["global_reward"]

    def reward(self, reward: float) -> tuple[tuple[float,], float]:
        return (reward,), reward


# Use this code to play the environment or for debugging purposes

# env = DiscreteMountainCar3Distance(gym.make("MountainCar-v0"))
# env.reset()
# print(env.step(0))

# play(env)
