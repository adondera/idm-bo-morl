from unicodedata import numeric
import matplotlib.pyplot as plt
import numpy as np

from dqn import DQN
from experiment import Experiment
from RND import RNDUncertainty


from n_sphere import n_sphere
import torch
from math import pi
import scipy.stats

from spherical_coords import reduce_dim, increase_dim


class BayesExperiment:
    def __init__(
        self,
        optimizer,
        utility,
        model,
        buffer,
        config_params,
        device,
        env,
        env_params,
        pbounds=(-pi, pi),
        alpha : np.array = None,
        metric_fun=lambda x: np.average(x[int(len(x) * 9 / 10) :]),
    ):
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
        self.pbounds = pbounds
        self.metric_fun = metric_fun


        self.numberPreferences = int(env_params["preferences"][0][0])

        if not isinstance(alpha, np.ndarray) or len(alpha) != self.numberPreferences:
            self.alpha = np.repeat(2.0, self.numberPreferences)
        else:
            self.alpha = alpha

        # TODO make into param
        self.prior = scipy.stats.dirichlet(alpha=self.alpha)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 5))

    def plot_bo(self, discarded_points = ([],[]), f=None):
        x = np.linspace(self.pbounds[0], self.pbounds[1], 4000)
        mean, sigma = self.optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
        self.ax.clear()
        if f:
            plt.plot(x, f(x))
        self.ax.plot(x, mean)
        # acq_max = self.optimizer
        self.ax.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
        self.ax.scatter(
            self.optimizer.space.params.flatten(),
            self.optimizer.space.target,
            c="red",
            s=50,
            zorder=10,
        )
        plt.draw()

    def run(self, number_of_experiments, discarded_experiments = 2, prior_only_experiments = 4):
        for experiment_id in range(number_of_experiments):

            learner = DQN(self.model, self.config_params, self.device, self.env)
            rnd = RNDUncertainty(
                self.uncertainty_scale,
                input_dim=self.env_params["states"][0][0],
                device=self.device,
            )
            
            # for the first burnout_experiments
            # next_preference_proj = sample from the prior
            # do not .register()

            prior_only_sample = experiment_id < prior_only_experiments
            discard_sample =  experiment_id < discarded_experiments
                
            if prior_only_sample:
                next_preference = self.prior.rvs(size=1).squeeze()
                next_preference_proj = reduce_dim(next_preference)
            else:
                next_preference_proj = self.optimizer.suggest(self.utility)

            self.plot_bo()

            next_preference = increase_dim(next_preference_proj)
            print("Next preference to probe is:", next_preference, " spherical: ", next_preference_proj)
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

            metric = self.metric_fun(global_rewards_experiment)
            self.global_rewards.append(metric)
            if not discard_sample:
                self.optimizer.register(params=next_preference_proj, target=metric)
            else:
                print("Discarding sample: ", next_preference_proj, metric)

            # self.optimizer.suggest(self.utility)
            self.plot_bo()
        plt.show(block=True)
