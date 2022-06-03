import matplotlib.pyplot as plt
import numpy as np

from dqn import DQN
from experiment import Experiment
from RND import RNDUncertainty


def reduce_dim(x):
    return x[:-1]


def add_dim(x: dict):
    l = []
    for _, it in x.items():
        l.append(it)
    l = np.array(l)
    sum = np.sum(l)
    return np.concatenate((l, [1.0 - sum]), dtype=np.single)


class BayesExperiment:
    def __init__(self, optimizer, utility, model, buffer, config_params, device, env, env_params):
        self.optimizer = optimizer
        self.utility = utility
        self.model = model
        self.buffer = buffer
        self.config_params = config_params
        self.device = device
        self.env = env
        self.env_params = env_params
        self.global_rewards = []
        self.uncertainty_scale = config_params.get("uncertainty_scale", 0)

        self.alpha = 0.1

        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 5))

    def plot_bo(self, f=None):
        x = np.linspace(0, 1, 4000)
        mean, sigma = self.optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
        self.ax.clear()
        if f:
            plt.plot(x, f(x))
        self.ax.plot(x, mean)
        self.ax.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
        self.ax.scatter(self.optimizer.space.params.flatten(), self.optimizer.space.target, c="red", s=50, zorder=10)
        plt.draw()

    def run(self, number_of_experiments):
        for _ in range(number_of_experiments):
            learner = DQN(self.model, self.config_params, self.device, self.env)
            rnd = RNDUncertainty(self.uncertainty_scale, input_dim=self.env_params["states"][0][0], device=self.device)
            next_preference_proj = self.optimizer.suggest(self.utility)
            # TODO make preferences work with dim!=2
            next_preference = add_dim(next_preference_proj)
            print("Next preference to probe is:", next_preference)
            experiment = Experiment(
                learner=learner,
                buffer=self.buffer,
                env=self.env,
                reward_dim=self.env_params["rewards"][0][0],
                preference=next_preference,
                params=self.config_params,
                device=self.device,
                uncertainty=rnd,
            )
            experiment.run()
            global_rewards_experiment = experiment.global_rewards
            metric = np.average(global_rewards_experiment)
            self.global_rewards.append(metric)
            self.optimizer.register(params=next_preference_proj, target=metric)
            self.optimizer.suggest(self.utility)
            self.plot_bo()
        plt.show(block=True)
