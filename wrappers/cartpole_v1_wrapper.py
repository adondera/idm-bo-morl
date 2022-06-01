from wrappers.rescaled_environment import RescaledEnv


class CartPoleV1AngleRewardWrapper(RescaledEnv):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top) 

    """

    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 1
        self.reward_names = ["abs(angle)"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        position, velocity, angle, angular_velocity = state
        R = abs(angle)
        z = reward
        return R, z
        # Returns (tuple of multi-objective rewards), z reward


class CartPoleV1AngleEnergyRewardWrapper(RescaledEnv):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top)
     - the opposite of the energy consumed 

    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_state = None
        self.numberPreferences = 2
        self.reward_names = ["1-angle", "-energy"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        previous_state = state if self.previous_state is None else self.previous_state
        position, velocity, angle, angular_velocity = state
        position_0, velocity_0, angle_0, angular_velocity_0 = previous_state
        delta_velocity = velocity - velocity_0
        energy = pow(delta_velocity, 2)
        R = (- abs(angle), -energy)

        z = reward
        self.previous_state = state
        return R, z
        # Returns (tuple of multi-objective rewards), z reward
