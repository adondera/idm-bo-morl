import gym
import torch
import matplotlib
import numpy as np

from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from config import default_params
from dqn import DQN
from replay_buffer import ReplayBuffer
import nobo, skopt
from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCarNormal
from experiment import Experiment
from config import default_params
from dqn import DQN
from RND import RNDUncertainty

matplotlib.use("Qt5agg")
# These configuration parameters need to be changed depending on the environment
config_params = default_params()
config_params["max_steps"] = int(2e5)
config_params["intrinsic_reward"] = True
config_params["k"] = 5

# env = DiscreteMountainCar3Distance(gym.make("MountainCar-v0"))
# env = PendulumRewardWrapper(gym.make("Pendulum-v1")) #this one doesn't work yet, because it has no env.action_space.n
# env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

env = DiscreteMountainCarNormal(gym.make("MountainCar-v0"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "reward_names": (env.__dict__["reward_names"], torch.StringType),
    "dones": ((1,), torch.bool),
}

preference = np.array([1.0], dtype=np.single)

model = torch.nn.Sequential(
    torch.nn.Linear(env_params["states"][0][0] + env_params["rewards"][0][0], 215),
    torch.nn.ReLU(),
    torch.nn.Linear(215, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 215),
    torch.nn.ReLU(),
    torch.nn.Linear(215, env.action_space.n),
)


def bayes_opt(input, output):
    bo = nobo.Optimizer([skopt.space.Real(0, 1), skopt.space.Real(0, 1)])
    for index in range(len(input)):
        x = input[index]
        y = output[index][0]
        bo.tell(x, y)
    x, lo, y, hi = bo.get_best()
    return np.array(x, dtype=np.single)


preferences = []
criterions = []
for i in range(100):
    if i >= 5:
        preference = bayes_opt(preferences, criterions)
    learner = DQN(model, config_params, device, env)
    buffer = ReplayBuffer(
        env_params, buffer_size=int(1e5), device=device, k=config_params.get("k", 1)
    )
    experiment = Experiment(
        learner=learner,
        buffer=buffer,
        env=env,
        reward_dim=env_params["rewards"][0][0],
        preference=preference,
        params=config_params,
        device=device,
    )
    experiment.run()
    preferences.append(preference)
    criterions.append(experiment.global_rewards)
learner = DQN(model, config_params, device, env)
rnd = RNDUncertainty(400, 2, device)
experiment = Experiment(
    learner=learner,
    buffer=buffer,
    env=env,
    reward_dim=env_params["rewards"][0][0],
    preference=preference,
    params=config_params,
    device=device,
    uncertainty=rnd,
)
experiment.run()
