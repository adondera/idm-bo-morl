from logging.config import dictConfig
import gym
import torch
import matplotlib
import numpy as np

from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from config import default_params
from dqn import DQN
from replay_buffer import ReplayBuffer
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
env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

# env = DiscreteMountainCarNormal(gym.make("MountainCar-v0"))

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


# TODO restructure code
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern

optimizer = BayesianOptimization(
    f=None,
    kernel=Matern(length_scale_bounds="fixed", nu=2.5),
    pbounds={"x": (0, 1)},
    verbose=2,
    random_state=1,
)
optimizer.set_gp_params(alpha=1.0)
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.1)

config_params["max_steps"] = int(1e4)


learner = DQN(model, config_params, device, env)
state_size = 4
rnd = RNDUncertainty(400, input_dim=state_size, device=device)
buffer = ReplayBuffer(
    env_params, buffer_size=int(1e5), device=device, k=config_params.get("k", 1)
)

import matplotlib.pyplot as plt


def plot_bo(bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    plt.figure(figsize=(16, 9))
    # plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()


def reduce_dim(x):
    return x[:-1]


def add_dim(x: dict):
    l = []
    for _, it in x.items():
        l.append(it)
    l = np.array(l)
    sum = np.sum(l)
    return np.concatenate((l, [1.0 - sum]), dtype=np.single)


global_rewards = []

continueExperiments = True
while continueExperiments:
    next_preference_proj = optimizer.suggest(utility)
    # TODO make preferences work with dim!=2
    next_preference = add_dim(next_preference_proj)
    print("Next preference to probe is:", next_preference)
    experiment = Experiment(
        learner=learner,
        buffer=buffer,
        env=env,
        reward_dim=env_params["rewards"][0][0],
        preference=next_preference,
        params=config_params,
        device=device,
        uncertainty=rnd,
    )
    experiment.run()
    global_rewards_experiment = experiment.global_rewards
    metric = np.average(global_rewards_experiment)
    global_rewards.append(metric)
    optimizer.register(params=next_preference_proj, target=metric)
    optimizer.suggest(utility)
    plot_bo(optimizer)

