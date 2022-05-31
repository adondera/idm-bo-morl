import torch as th


class RNDUncertainty:
    """ This class uses Random Network Distillation to estimate the uncertainty/novelty of states. """

    def __init__(self, scale, input_dim, device, hidden_dim=1024, embed_dim=256, **kwargs):
        self.scale = scale
        self.criterion = th.nn.MSELoss(reduction='none')
        # YOUR CODE HERE
        self.target_net = th.nn.Sequential(th.nn.Linear(input_dim, hidden_dim), th.nn.ReLU(),
                                           th.nn.Linear(hidden_dim, hidden_dim), th.nn.ReLU(),
                                           th.nn.Linear(hidden_dim, embed_dim))
        self.predict_net = th.nn.Sequential(th.nn.Linear(input_dim, hidden_dim), th.nn.ReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.ReLU(),
                                            th.nn.Linear(hidden_dim, embed_dim))
        self.target_net.to(device)
        self.predict_net.to(device)
        self.optimizer = th.optim.Adam(self.predict_net.parameters())

    def error(self, state, preference):
        """ Computes the error between the prediction and target network. """
        if not isinstance(state, th.Tensor):
            state = th.tensor(state)
        if len(state.shape) == 1:
            state.unsqueeze(dim=0)
        # processed_input = th.cat((state, preference), dim=-1)
        processed_input = state
        # YOUR CODE HERE: return the RND error
        return self.criterion(self.predict_net(processed_input), self.target_net(processed_input))

    def observe(self, state, preference, **kwargs):
        """ Observes state(s) and 'remembers' them using Random Network Distillation"""
        # YOUR CODE HERE
        self.optimizer.zero_grad()
        self.error(state, preference).mean().backward()
        self.optimizer.step()

    def __call__(self, state, preference, **kwargs):
        """ Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor. """
        # YOUR CODE HERE
        return self.scale * self.error(state, preference).mean(dim=-1)
