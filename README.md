
## Instructions

Run `bayes_main.py` for experiments involving the bayesian optimization process.
Run `dqn_main.py` to test the DDQN agent in a single experiment on the environment

#### Parameters

Each environment requires some sort of parameter tuning.
Here are some potentially good configurations based on trial and error. They could probably be improved.

More information about each configuration parameter can be found in `config.py`, along with their default values.
* CartPole
  * `max_steps` = `1e5`
  * `grad_repeats` = `1`
  * `uncertainty_scale` = `0` or `intrinsic_reward` set to `False`
  * If doing Bayesian Optimization it may be a good idea to turn off environment rendering with `render_step = 0`
* MountainCar
  * `max_steps` = `2e5`. If computation resources are an issue, a lower value should also work.
  * `grad_repeats` = `10`. If computation resources are an issue, a lower value should also work.
  * `intrinsic_reward` = `True`  `uncertainty_scale` = `400`. Very important parameter for this environment. 
