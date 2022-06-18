import random
from .mo_wrapper import MOWrapper
import config

# Change these two to tweak the intrinsic rewards with RND. The higher the uncertainty scale,
# the more important intrinsic rewards will be. Depends from environment to environment, requires tweaking.
def get_cartpole_config():
    config_params = config.default_params()
    config_params["k"] = 10
    config_params[
        "render_step"
    ] = 0
    config_params["intrinsic_reward"] = False
    config_params["uncertainty_scale"] = 0
    config_params["grad_repeats"] = int(1)
    config_params['max_steps'] = int(2E5)
    config_params['max_episodes'] = int(1e3)
    return config_params


class CartPoleConstRewardWrapper(MOWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top)

    """

    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 2
        self.reward_names = ["const (1)", "const (2)"]
        self.tags = ["CartPole", "Constant 1 / Constant 2"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        position, velocity, angle, angular_velocity = state
        R = abs(angle)
        z = reward
        return R, z
        # Returns (tuple of multi-objective rewards), z reward

    def get_config(self):
        return get_cartpole_config()


class CartPoleNoisyRewardWrapper(MOWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top)

    """

    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 2
        self.reward_names = ["-Angle", "Noisy sensor"]
        self.tags = ["CartPole", "Negative Angle / Noisy Random"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        position, velocity, angle, angular_velocity = state
        R = (-abs(angle), -random.random() / 2)
        z = reward
        return R, z
        # Returns (tuple of multi-objective rewards), z reward

    def get_config(self):
        return get_cartpole_config()


class CartPoleV1AngleRewardWrapper(MOWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top) 

    """

    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 1
        self.reward_names = ["Abs(angle)"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        position, velocity, angle, angular_velocity = state
        R = abs(angle)
        z = reward
        return R, z
        # Returns (tuple of multi-objective rewards), z reward

    def get_config(self):
        return get_cartpole_config()


class CartPoleV1AngleNegEnergyRewardWrapper(MOWrapper):
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
        self.reward_names = ["-Angle", "-Energy"]
        self.tags = ["CartPole", "Negative Angle / Negative Velocity"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        previous_state = state if self.previous_state is None else self.previous_state
        position, velocity, angle, angular_velocity = state
        position_0, velocity_0, angle_0, angular_velocity_0 = previous_state
        delta_velocity = velocity - velocity_0
        energy = pow(delta_velocity, 2)
        R = (-abs(angle), -abs(velocity))

        z = reward
        self.previous_state = state
        return R, z
        # Returns (tuple of multi-objective rewards), z reward

    def reset(self, **kwargs):
        self.previous_state = None
        return self.env.reset(**kwargs)

    def get_config(self):
        return get_cartpole_config()


class CartPoleV1AnglePosEnergyRewardWrapper(MOWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is:
     - the absolute value of the angle (measured from the top)
     - the energy consumed

    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_state = None
        self.numberPreferences = 2
        self.reward_names = ["-Angle", "Energy"]
        self.tags = ["CartPole", "Negative Angle / Positive Velocity"]

    def reward(self, reward):
        base_env = self.env.env
        state = base_env.state
        previous_state = state if self.previous_state is None else self.previous_state
        position, velocity, angle, angular_velocity = state
        position_0, velocity_0, angle_0, angular_velocity_0 = previous_state
        delta_velocity = velocity - velocity_0
        energy = pow(delta_velocity, 2)
        R = (-abs(angle), abs(velocity))

        z = reward
        self.previous_state = state
        return R, z
        # Returns (tuple of multi-objective rewards), z reward

    def reset(self, **kwargs):
        self.previous_state = None
        return self.env.reset(**kwargs)

    def get_config(self):
        return get_cartpole_config()


class SparseCartpole(MOWrapper):
    """
    Usage:
        env = SparseCartpole(CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1")))
    """

    total_steps = 0

    def __init__(self, env: MOWrapper, steps_target=200):
        super().__init__(env)
        self.steps_target = steps_target
        self.numberPreferences = self.env.numberPreferences
        self.reward_names = self.env.reward_names
        self.tags = self.env.tags + ["Sparse"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.total_steps += 1
        R, z = self.reward(reward)
        return observation, (R, z), done, info

    def reward(self, reward):
        R, _ = self.env.reward(reward)
        z = 0 if self.total_steps < self.steps_target else 1
        return R, z

    def reset(self, **kwargs):
        self.total_steps = 0
        self.previous_state = None
        return self.env.reset(**kwargs)

    def get_config(self):
        return get_cartpole_config()
