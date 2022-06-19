import gym
import torch
import matplotlib
from sklearn.gaussian_process.kernels import Matern

from wrappers.cartpole_v1_wrapper import SparseCartpole, CartPoleNoisyRewardWrapper
from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCarVelocityDistance
from wrappers.mo_wrapper import RescaledReward
from replay_buffer import ReplayBuffer
from config import default_params
from bayes_experiment import BayesExperiment
from BayesianOptimization.bayes_opt import BayesianOptimization, UtilityFunction

from math import pi
import numpy as np
import os

if os.environ.get("DESKTOP_SESSION") == "i3":
    matplotlib.use("tkagg")
else:
    matplotlib.use("Qt5agg")

# env = RescaledReward(SparseCartpole(CartPoleNoisyRewardWrapper(gym.make("CartPole-v1"))))
env = RescaledReward(
    DiscreteMountainCarVelocityDistance(gym.make("MountainCar-v0")), [10, 1]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "reward_names": (env.__dict__["reward_names"], torch.StringType),
    "dones": ((1,), torch.bool),
}

model = torch.nn.Sequential(
    torch.nn.Linear(env_params["states"][0][0] + env_params["rewards"][0][0], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, env.action_space.n),
)

config_params = env.get_config()
config_params["preference_dim"] = env_params["preferences"][0][0]
"""
    `discarded_experiments_length_factor` = n will make the initial experiments n times longer
"""
config_params["discarded_experiments_length_factor"] = 1.0

number_BO_episodes = config_params["number_BO_episodes"]
config_params["discarded_experiments"] = 0  # max(2,number_BO_episodes/10)
config_params["prior_only_experiments"] = 0  # max(4, number_BO_episodes/5)

# --- These parameters affect the bayesian optimization process ---

# to try and optimize with negative weights, change to
#  default_bounds = (-pi, pi)
default_bounds = (0, pi / 2)
dirichlet_alpha = np.array(
    config_params.get(
        "dirichlet_alpha", np.repeat(5.0, int(env_params["preferences"][0][0] - 1))
    )
)

alpha = config_params["alpha"]
length_scale = default_bounds[1] * config_params["length_scale_to_bounds_ratio"]
# xi should be the ~half the difference between the lowest and highest score
xi = config_params["xi"]
kappa = config_params["kappa"]
nu = config_params["nu"]
utility_function = config_params.get("utility_function", "ucb")

optimizer = BayesianOptimization(
    f=None,
    kernel=Matern(length_scale_bounds="fixed", length_scale=length_scale, nu=nu),
    pbounds=([default_bounds] * int(env_params["preferences"][0][0] - 1)),
    verbose=2,
    random_state=1,
)
utility = UtilityFunction(kind=utility_function, kappa=kappa, xi=xi)
optimizer.set_gp_params(alpha=alpha)

buffer = ReplayBuffer(
    env_params, buffer_size=int(1e5), device=device, k=config_params.get("k", 1)
)

bayes_experiment = BayesExperiment(
    optimizer=optimizer,
    utility=utility,
    model=model,
    buffer=buffer,
    config_params=config_params,
    device=device,
    env=env,
    env_params=env_params,
    pbounds=default_bounds,
    dirichlet_alpha=dirichlet_alpha,
)

bayes_experiment.run(number_BO_episodes)

evaluation = bayes_experiment.evaluate_best_preference(num_samples=5)
average_global_reward = np.average([metric for _, metric, _ in evaluation])

print("Average global reward:", average_global_reward)
