import gym
from gym.envs.classic_control.pendulum import angle_normalize


class PendulumRewardWrapper(gym.Wrapper):
    def __init__(self, env, scale=None):
        super().__init__(env)
        self.numberPreferences = 4
        self.reward_names = ["angle", "angular_velocity", "torque", "energy"]
        self.scale = scale
        if scale is None:
            self.scale = [1] * self.env.numberPreferences

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        (R, z) = self.reward(reward, action)
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info

    def reward(self, reward, action):
        base_env = self.env.env
        state = base_env.state
        th, thdot = state
        angle_cost = angle_normalize(th) ** 2
        angular_velocity_cost = 0.1 * thdot ** 2
        action_cost = 0.001 * (base_env.last_u ** 2)

        energy_cost = abs(base_env.last_u * thdot)  # should also include dt
        R = {
            "angle": -angle_cost,
            "angular_velocity": -angular_velocity_cost,
            "torque": -action_cost,
            "energy": -energy_cost,
        }
        z = reward
        return R, z  # Returns (tuple of multi-objective rewards), z reward


class PendulumEnergyRewardWrapper(PendulumRewardWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is: 
        - Angle from the top
        - Energy (abs of the product between torque and angular velocity)
    """

    def __init__(self, env, scale=None):
        super().__init__(env, scale)
        self.numberPreferences = 2
        self.reward_names = ["angle", "energy"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        R_0, z = self.reward(reward, action)
        R = (R_0["angle"], R_0["energy"])
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info


class PendulumMultiObjectiveOriginalRewardWrapper(PendulumRewardWrapper):
    """
    Usage: env = CartPoleV1AngleRewardWrapper(gym.make("CartPole-v1"))

    The reward is: 
        - Angle from the top
        - Angular velocity
        - Applied torque (action)
    """

    def __init__(self, env, scale=None):
        super().__init__(env, scale)
        self.numberPreferences = 3
        self.reward_names = ["angle", "angular_velocity"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        R_0, z = self.reward(reward, action)
        R = (R_0["angle"], R_0["angular_velocity"], R_0["torque"])
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info
