from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern, RBF, Exponentiation, ExpSineSquared


import matplotlib.pyplot as plt
import numpy as np


def plot_bo(bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    plt.figure(figsize=(16, 9))
    # plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()


def black_box_function(x):
    y = -(x ** 2) + 3
    return y


optimizer = BayesianOptimization(
    f=None,
    kernel=Matern(length_scale_bounds="fixed", nu=2.5),
    pbounds={"x": (-2, 2)},
    verbose=2,
    random_state=1,
)
optimizer.set_gp_params(alpha=1.0)

from bayes_opt import UtilityFunction

# ucb doesn't use xi, only kappa
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.1)

next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)

target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)


for _ in range(2):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=1.2, target=target - 5.0)

    print(target, next_point)
    plot_bo(optimizer)

for _ in range(5):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    print(target, next_point)
    plot_bo(optimizer)

for _ in range(3):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=1.2, target=25.0 + target)

    print(target, next_point)
    plot_bo(optimizer)

print(optimizer.max)

