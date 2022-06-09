from wrappers.mo_wrapper import MOWrapper

FPS = 50
SCALE = 30.0
LEG_DOWN = 18
VIEWPORT_W = 600
VIEWPORT_H = 400


class LunarLanderVelocityDistanceWrapper(MOWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.numberPreferences = 2
        self.reward_names = ["-Distance to goal", "-Velocity"]

    def reward(self, reward):
        """
        :param reward: The reward sampled from the environment
        :return: A tuple containing the multi-objective reward (r) and the environment reward (z)
        The multi-objective reward contains the distance to the goal and the negative of the velocity of the agent
        """
        pos = self.env.unwrapped.lander.position
        vel = self.env.unwrapped.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        # TODO: Figure out the distance to the helipad
        return (-state[2], -state[3],), reward
