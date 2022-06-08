class RescaleWrapper:

    def __init__(self, env, scale=None):
        self.env = env
        if scale is None:
            scale = [1] * self.env.numberPreferences
        self.scale = scale

    def step(self, action):
        observation, (R, z), done, info = self.env.step(action)
        R = [x * self.scale[i] for i, x in enumerate(R)]
        return observation, (R, z), done, info
