import sys
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import ensemble

from torch.distributions import Normal
from kernels import batch_rbf
from normalizer import TransitionNormalizer

i=0
j=0

class Model(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, d_action, d_state, n_hidden, n_layers, ensemble_size, non_linearity='leaky_relu', device=torch.device('cuda')):

        super().__init__()

        self.hypergan = ensemble.HyperGAN(device, d_action, d_state)
        # match_hypergan(self.hypergan, ensemble_size, d_action, d_state, n_hidden, device)
        self.normalizer = None

        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.device = device
        
        self._fetch_ensemble()  
        
        self.to(device)

    
    def _fetch_ensemble(self):
        codes = torch.randn(5, self.ensemble_size, 64).to(self.device)
        self.ensemble = self.hypergan.generator(codes)
        

    def setup_normalizer(self, normalizer):
        self.normalizer = TransitionNormalizer()
        self.normalizer.set_state(normalizer.get_state())

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is None:
            return states, actions

        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is None:
            return state_deltas

        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, var):
        # denormalize to return in raw state space
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            var = self.normalizer.denormalize_state_delta_vars(var)
        return delta_mean, var

    def _propagate_network(self, states, actions, return_preds=None, for_loss=False):
        self._fetch_ensemble()
        inp = torch.cat((states, actions), dim=2)
        means, stds, preds = [], [], []
 
        if inp.shape[1] == 1: # utility measurement
            preds = [self.hypergan.eval_f(layer, inp[i]) for (i, layer) in enumerate(zip(*self.ensemble))]
            preds = torch.stack(preds).view(-1, self.d_state)
            mean = preds.mean(0).unsqueeze(0)
            if for_loss is True:
                return mean
            else:
                var = preds.var(0).unsqueeze(0)
                return mean, var

        if not return_preds:
            if for_loss is True:
                for batch_set in inp: ## batch x states
                    chunks = torch.stack([self.hypergan.eval_f(layer, batch_set) for layer in zip(*self.ensemble)])
                    means.append(chunks)
                delta_mean = torch.stack(means).mean(0) ## ensemble size x batch size x d_state
                return delta_mean
            else:
                for batch_set in inp: ## batch x states
                    chunks = torch.stack([self.hypergan.eval_f(layer, batch_set) for layer in zip(*self.ensemble)])
                    means.append(chunks)
                    stds.append(chunks)

                delta_mean = torch.stack(means) ## ensemble size x batch size x d_state
                delta_mean = delta_mean.mean(0)
                var = torch.stack(stds) ## ensemble size x batch size x d_state
                var = var.var(0)
                return delta_mean, var

        else:
            preds = [self.hypergan.eval_f(layer, inp[i]) for (i, layer) in enumerate(zip(*self.ensemble))]
            return torch.stack(preds)

    def forward(self, states, actions):
        """
        predict next state mean and variance.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)

        Returns:
            next state means (torch tensor): (ensemble_size, batch size, dim_state)
            next state variances (torch tensor): (ensemble_size, batch size, dim_state)
        """

        normalized_states, normalized_actions = self._pre_process_model_inputs(states, actions)
        normalized_delta_mean, normalized_var = self._propagate_network(normalized_states, normalized_actions)
        delta_mean, var = self._post_process_model_outputs(normalized_delta_mean, normalized_var)
        next_state_mean = delta_mean + states.to(self.device)
        return next_state_mean, var

    def forward_all(self, states, actions):
        """
        predict next state mean and variance of a batch of states and actions for all models.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch tensor): (batch size, dim_state)
            actions (torch tensor): (batch size, dim_action)

        Returns:
            next state means (torch tensor): (batch size, ensemble_size, dim_state)
            next state variances (torch tensor): (batch size, ensemble_size, dim_state)
        """
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_state_means, next_state_vars = self(states, actions)
        next_state_means = next_state_means.transpose(0, 1)
        next_state_vars = next_state_vars.transpose(0, 1)
        return next_state_means, next_state_vars

    def sample(self, mean, var):
        """
        sample next state, given next state mean and variance

        Args:
            mean (torch tensor): any shape
            var (torch tensor): any shape

        Returns:
            next state (torch tensor): same shape as inputs
        """

        return Normal(mean, torch.sqrt(var)).sample()

    """ Amortized Stein Variational Gradient Descent Ops """
    def svgd_batch_vectorized(self, means_zj, targets):
        alpha = 1e-3
        svgd_sum, loss, kappa_sum = 0, 0, 0
        
        log_probs = F.mse_loss(means_zj, targets)  # calculate log probs
        log_probs.backward(retain_graph=True)
        logp_z = autograd.grad(log_probs.sum(), means_zj)[0]  # [particles, batch, d_output]
        layers = [] # put the ensemble into a sensible list of layers
        for item in self.ensemble:
            item = item.view(self.ensemble_size, -1)
            layers.append(item)
        layers = torch.cat(layers, dim=-1).view(self.ensemble_size, -1)  # [particles, params]

        eps_svgd = torch.ones_like(layers).uniform_(-1e-7, 1e-7)  # protect against two identical inputs
        
        kappa = batch_rbf(layers + eps_svgd)  # [particles, particles]
        grad_kappa = autograd.grad(kappa.sum(), layers)[0]  # [particles, params]
        
        logp_z = logp_z.transpose(0, 1).unsqueeze(2)  # [batch, particles, 1, d_output]
        logp_z = logp_z.mean(0).mean(-1)  # [particles, 1]
        kernel_logp = torch.matmul(kappa.detach(), logp_z)  # [particles, 1]

        svgd = (kernel_logp + alpha * grad_kappa) / self.ensemble_size  # [particles, params]
        
        autograd.backward(layers, grad_tensors=svgd)
        
        svgd_val = svgd.mean().item()
        kappa_val = kappa.mean().item()
        log_probs_val = log_probs.mean()
        
        return log_probs_val, kappa_val, svgd_val




    def loss(self, states, actions, state_deltas, training_noise_stdev=0):
        """
        compute loss given states, actions and state_deltas
        Stein gradient \delta z
        Its the expectation over z drawn from zi : particles
            [the gradient wrt z -- of [log p(z) * k(z, zi)]
            + [gradient wrt z -- of k(z, zi)]
        First term is standard likelihood
        Second term is repulsive

        """

        states, actions = self._pre_process_model_inputs(states, actions)
        targets = self._pre_process_model_targets(state_deltas)

        if not np.allclose(training_noise_stdev, 0):
            states += torch.randn_like(states) * training_noise_stdev
            actions += torch.randn_like(actions) * training_noise_stdev
            targets += torch.randn_like(targets) * training_noise_stdev

        means_zj = self._propagate_network(states, actions, return_preds=False, for_loss=True)
        
        loss, kappa, svgd_val = self.svgd_batch_vectorized(means_zj, targets)
        return loss, kappa, svgd_val


    def likelihood(self, states, actions, next_states):
        """
        input raw (un-normalized) states, actions and state_deltas

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)
            next_states (torch tensor): (ensemble_size, batch size, dim_state)

        Returns:
            likelihood (torch tensor): (batch size)
        """

        next_states = next_states.to(self.device)

        with torch.no_grad():
            mu, var = self(states, actions)     # next state and variance

        pdf = Normal(mu, torch.sqrt(var))
        log_likelihood = pdf.log_prob(next_states)

        log_likelihood = log_likelihood.mean(dim=2).mean(dim=0)     # mean over all state components and models

        return log_likelihood
