from wrappers.rescaled_environment import RescaledEnv


class DiscreteMountainCar3Distance(RescaledEnv):
    def __init__(self, env, max_episode_length=None):
        super().__init__(env, max_episode_length)
        self.numberPreferences = 3
        self.reward_names = ["-Distance to left hill", "-Distance to start", "-Distance to right hill"]
        self.tags = ["Sparse", "MountainCar", "Distance left / Distance center / Distance right"]

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
    def __init__(self, env, max_episode_length=None):
        super().__init__(env, max_episode_length)
        self.numberPreferences = 1
        self.reward_names = ["Current velocity"]
        self.tags = ["Sparse", "MountainCar", "Velocity"]

    def reward(self, reward: float) -> tuple[tuple[float,], float]:
        """
        :param reward: The reward sampled from the environment (-1 or 0 if it reaches the goal)
        :return: A tuple containing the multi-objective reward (r) and the environment reward (z)
        The multi-objective reward contains only the velocity of the agent
        """
        current_position, current_velocity = self.env.unwrapped.state
        return (current_velocity,), reward


class DiscreteMountainCarVelocityDistance(RescaledEnv):
    def __init__(self, env, max_episode_length=None):
        super().__init__(env, max_episode_length)
        self.numberPreferences = 2
        self.reward_names = ["Velocity", "-Distance to goal"]
        self.tags = ["Sparse", "MountainCar", "Velocity / Negative distance"]

    def reward(self, reward: float) -> tuple[tuple[float, float], float]:
        goal_position = self.env.unwrapped.goal_position
        start_position = -0.5
        current_position, current_velocity = self.env.unwrapped.state
        distance_right_to_start = abs(goal_position - start_position)
        distance_to_right_hill = abs(current_position - goal_position)
        distance_metric = min(distance_right_to_start, distance_to_right_hill)
        return (abs(current_velocity), -distance_metric), reward
