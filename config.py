def default_params():
    """These are the default parameters used int eh framework."""
    return {  # Debugging outputs and plotting during training
        "plot_frequency": 10,  # plots a debug message avery n steps
        "plot_train_samples": True,  # whether the x-axis is env.steps (True) or episodes (False)
        "print_when_plot": True,  # prints debug message if True
        "print_dots": False,  # prints dots for every gradient update
        # Environment parameters
        "env": "CartPole-v0",  # the environment the agent is learning in
        "run_steps": 0,  # samples whole episodes if run_steps <= 0
        "max_episode_length": 200,  # maximum number of steps per episode
        # Runner parameters
        "max_episodes": int(1e6),  # experiment stops after this many episodes
        "max_steps": int(2e5),  # experiment stops after this many steps
        "multi_runner": False,  # uses multiple runners if True
        "parallel_environments": 4,  # number of parallel runners  (only if multi_runner==True)
        # Exploration parameters
        "epsilon_anneal_time": int(
            2e4
        ),  # exploration anneals epsilon over these many steps
        "epsilon_finish": 0.05,  # annealing stops at (and keeps) this epsilon
        "epsilon_start": 1,  # annealing starts at this epsilon
        # Optimization parameters
        "lr": 1e-4,  # 5E-4,                       # learning rate of optimizer
        "gamma": 0.99,  # discount factor gamma
        "batch_size": 2048,  # number of transitions in a mini-batch
        "grad_norm_clip": 1,  # gradent clipping if grad norm is larger than this
        # DQN parameters
        "replay_buffer_size": int(
            1e5
        ),  # the number of transitions in the replay buffer
        "use_last_episode": False,  # whether the last episode is always sampled from the buffer
        "target_model": True,  # whether a target model is used in DQN
        "target_update": "soft",  # 'soft' target update or hard update by regular 'copy'
        "target_update_interval": 10,  # interval for the 'copy' target update
        "soft_target_update_param": 0.01,  # update parameter for the 'soft' target update
        "double_q": True,  # whether DQN uses double Q-learning
        "grad_repeats": 10,  # how many gradient updates / runner call
        # RND parameters
        "intrinsic_reward": True,  # Whether we use intrinsic rewards (RND) or not
        "uncertainty_scale": 10,  # Factor with which to scale the uncertainty reward
        "preference_dim": 0,  # Dimension of preferences. Default value of 0 means preference is not taken into account
        # MO parameters
        "k": 5,  # number of preferences that are sampled in her storing
        "multi_objective": True,  # whether to use multi or single-objective RL
        "norm": 1,  # The power at which each objective reward is raised (e.g. 2 ==> quadratic rewards)
        # Image input parameters
        "pixel_observations": False,  # use pixel observations (we will not use this feature here)
        "pixel_resolution": (78, 78),  # scale image to this resoluton
        "pixel_grayscale": True,  # convert image into grayscale
        "pixel_add_last_obs": True,  # stacks 2 observations
        "pixel_last_obs_delay": 3,  # delay between the two stacked observations
        "render_step": 100,  # Number of episodes between different renders. Set to <=0 to disable rendering
        # Bayes optimization parameters
        # "dirichelet_alpha": [5.0],
        "alpha": 0.1,
        "length_scale_to_bounds_ratio": 0.5,
        "xi": 100,
        "kappa": 10.0,
        "nu": 2.5,
        "utility_function": "ucb",
        "number_BO_episodes": 5,
        "wandb": True,
    }
