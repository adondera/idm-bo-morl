import torch
from copy import deepcopy


class DQN:
    def __init__(self, model, params):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.get('lr', 5E-4))
        self.gamma = params.get('gamma', 0.99)
        self.criterion = torch.nn.MSELoss()
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.target_model = deepcopy(model)
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.soft_target_update_param = params.get('soft_target_update_param', 0.1)
        self.double_q = params.get('double_q', True)
        self.all_parameters = model.parameters()

    @staticmethod
    def _process_input(states, preferences):
        processed_input = torch.cat((states, preferences), dim=-1)
        return processed_input

    def target_model_update(self):
        """ This function updates the target network. """
        if self.target_model is not None:
            for tp, mp in zip(self.target_model.parameters(), self.model.parameters()):
                tp *= (1 - self.soft_target_update_param)
                tp += self.soft_target_update_param * mp.detach()

    def q_values(self, states, preferences, target=False):
        """ Returns the Q-values of the given "states". Uses the target network if "target=True". """
        target = target and self.target_model is not None
        processed_input = DQN._process_input(states, preferences)
        return (self.target_model if target else self.model)(processed_input)

    def _current_values(self, batch):
        """ Computes the Q-values of the 'states' and 'actions' of the given "batch". """
        qvalues = self.q_values(batch['states'], batch['preferences'])
        return qvalues.gather(dim=-1, index=batch['actions'])

    def _next_state_values(self, batch):
        """ Computes the Q-values of the 'next_state' of the given "batch".
            Is greedy w.r.t. to current Q-network or target-network, depending on parameters. """
        with torch.no_grad():  # Next state values do not need gradients in DQN
            # Compute the next states values (with target or current network)
            qvalues = self.q_values(batch['next_states'], batch['preferences'], target=True)
            # Compute the maximum (note the case of double Q-learning)
            if self.target_model is None or not self.double_q:
                # If we do not do double Q-learning or if there is no target network
                qvalues = qvalues.max(dim=-1, keepdim=True)[0]
            else:
                # If we do double Q-learning
                next_values = self.q_values(batch['next_states'], batch['preferences'], target=False)
                actions = next_values.max(dim=-1)[1].unsqueeze(dim=-1)
                qvalues = qvalues.gather(dim=-1, index=actions)
            return qvalues

    def train(self, batch):
        """ Performs one gradient decent step of DQN. """
        self.model.train(True)
        # Compute TD-loss. We multiply with the preferences here to obtain a single reward.
        targets = torch.sum(batch['rewards'] * batch['preferences'], dim=-1).unsqueeze(-1) + self.gamma * (
                ~batch['dones'] * self._next_state_values(batch))
        loss = self.criterion(self._current_values(batch), targets.detach())
        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
        self.optimizer.step()
        # Update target network (if specified) and return loss
        self.target_model_update()
        return loss.item()
