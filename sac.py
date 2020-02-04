import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torch.distributions import Normal
from distributions import TanhNormal


i=0
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

def copy_tensor(x):
    return x.clone().detach() #.cpu()


def danger_mask(x):
    mask = torch.isnan(x) + torch.isinf(x)
    mask = torch.sum(mask, dim=1) > 0
    return mask



def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


class Replay:
    def __init__(self, d_state, d_action, size):
        self.states = torch.zeros([size, d_state]).float()
        self.next_states = torch.zeros([size, d_state]).float()
        self.actions = torch.zeros([size, d_action]).float()
        self.rewards = torch.zeros([size, 1]).float()
        self.masks = torch.zeros([size, 1]).float()
        self.ptr = 0

        self.d_state = d_state
        self.d_action = d_action
        self.size = size

        self.normalizer = None
        self.buffer_full = False

    def clear(self):
        d_state = self.d_state
        d_action = self.d_action
        size = self.size
        self.states = torch.zeros([size, d_state]).float()
        self.next_states = torch.zeros([size, d_state]).float()
        self.actions = torch.zeros([size, d_action]).float()
        self.rewards = torch.zeros([size, 1]).float()
        self.masks = torch.zeros([size, 1]).float()
        self.ptr = 0
        self.buffer_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def add(self, states, actions, rewards, next_states, masks=None):
        n_samples = states.size(0)

        if masks is None:
            masks = torch.ones(n_samples, 1)

        states, actions, rewards, next_states = copy_tensor(states), copy_tensor(actions), copy_tensor(rewards), copy_tensor(next_states)
        rewards = rewards.unsqueeze(1)
        
        # skip ones with NaNs and Infs
        # print (danger_mask.shape, states.shape, actions.shape, rewards.shape, next_states.shape
        skip_mask = danger_mask(states) + danger_mask(actions) + danger_mask(rewards) + danger_mask(next_states)
        include_mask = (skip_mask == 0)

        n_samples = torch.sum(include_mask).item()
        if self.ptr + n_samples >= self.size:
            # crude, but ok
            self.ptr = 0
            self.buffer_full = True

        i = self.ptr
        j = self.ptr + n_samples

        self.states[i:j] = states[include_mask]
        self.actions[i:j] = actions[include_mask]
        self.rewards[i:j] = rewards[include_mask]
        self.next_states[i:j] = next_states[include_mask]
        self.masks[i:j] = masks

        self.ptr = j

    def sample(self, batch_size):
        idxs = np.random.randint(len(self), size=batch_size)
        states, actions, rewards, next_states, masks = self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.masks[idxs]
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        return states, actions, rewards, next_states, masks

    def __len__(self):
        if self.buffer_full:
            return self.size
        return self.ptr


def init_weights(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)


def init_weights_rlkit(layer):
    init_w = 1e-3
    nn.init.uniform_(layer.weight, -init_w, init_w)
    nn.init.uniform_(layer.bias, -init_w, init_w) 


class ParallelLinear_rlkit(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, final_layer=False):
        super().__init__()

        weights = []
        biases = []
        for j in range(ensemble_size):
            weight = torch.Tensor(n_in, n_out).float()
            bias = torch.Tensor(1, n_out).float()
            if final_layer:
                if j == 0:
                    nn.init.uniform_(weight, -3e-3, 3e-3)  # mean
                    nn.init.uniform_(bias, -3e-3, 3e-3)  # mean
                if j == 1:
                    nn.init.uniform_(weight, -1e-3, 1e-3) # log std
                    nn.init.uniform_(bias, -1e-3, 1e-3)  # log std
            else:
                fanin_init(weight)
                bias.fill_(0.1)

            weights.append(weight)
            biases.append(bias)

        weights = torch.stack(weights)
        biases = torch.stack(biases)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, self.weights)
        return op


class ParallelLinear(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size):
        super().__init__()

        weights = []
        biases = []
        for _ in range(ensemble_size):
            weight = torch.Tensor(n_in, n_out).float()
            bias = torch.Tensor(1, n_out).float()
            nn.init.orthogonal_(weight)
            bias.fill_(0.0)

            weights.append(weight)
            biases.append(bias)

        weights = torch.stack(weights)
        biases = torch.stack(biases)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, self.weights)
        return op


class ActionValueFunction_rlkit(nn.Module):
    def __init__(self, d_state, d_action, n_hidden):
        super().__init__()
        self.layers = nn.Sequential(ParallelLinear_rlkit(d_state + d_action, n_hidden, ensemble_size=2),
                                    nn.ReLU(),
                                    ParallelLinear_rlkit(n_hidden, n_hidden, ensemble_size=2),
                                    nn.ReLU(),
                                    ParallelLinear_rlkit(n_hidden, 1, ensemble_size=2, final_layer=True))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = x.unsqueeze(0).repeat(2, 1, 1)
        y1, y2 = self.layers(x)
        return y1, y2


class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_hidden):
        super().__init__()
        self.layers = nn.Sequential(ParallelLinear(d_state + d_action, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, n_hidden, ensemble_size=2),
                                    nn.LeakyReLU(),
                                    ParallelLinear(n_hidden, 1, ensemble_size=2))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = x.unsqueeze(0).repeat(2, 1, 1)
        y1, y2 = self.layers(x)
        return y1, y2


class StateValueFunction_rlkit(nn.Module):
    def __init__(self, d_state, n_hidden):
        super().__init__()
        init_w = 3e-3
        one = nn.Linear(d_state, n_hidden)
        fanin_init(one.weight)
        one.bias.data.fill_(0.1)
        
        two = nn.Linear(n_hidden, n_hidden)
        fanin_init(two.weight)
        two.bias.data.fill_(0.1)
        
        three = nn.Linear(n_hidden, 1)
        three.weight.data.uniform_(-init_w, init_w)
        three.bias.data.uniform_(-init_w, init_w)

        self.layers = nn.Sequential(one,
                                    nn.ReLU(),
                                    two,
                                    nn.ReLU(),
                                    three)

    def forward(self, state):
        result = self.layers(state)
        return result


class StateValueFunction(nn.Module):
    def __init__(self, d_state, n_hidden):
        super().__init__()

        one = nn.Linear(d_state, n_hidden)
        init_weights(one)
        two = nn.Linear(n_hidden, n_hidden)
        init_weights(two)
        three = nn.Linear(n_hidden, 1)
        init_weights(three)

        self.layers = nn.Sequential(one,
                                    nn.LeakyReLU(),
                                    two,
                                    nn.LeakyReLU(),
                                    three)

    def forward(self, state):
        result = self.layers(state)
        return result


class TanhGaussianPolicy(nn.Module):
    def __init__(self, d_state, d_action, n_hidden):
        super().__init__()

        one = nn.Linear(d_state, n_hidden)
        one.bias.data.fill_(0.1)
        fanin_init(one.weight)

        two = nn.Linear(n_hidden, n_hidden)
        two.bias.data.fill_(0.1)
        fanin_init(two.weight)

        three = nn.Linear(n_hidden, 2 * d_action)
        init_weights_rlkit(three)

        self.layers = nn.Sequential(one,
                                    nn.ReLU(),
                                    two,
                                    nn.ReLU(),
                                    three)

    def forward(self, state):
        y = self.layers(state)
        mu, log_std = torch.split(y, y.size(1) // 2, dim=1)

        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # normal = Normal(mu, std)
        # pi = normal.rsample()           # with re-parameterization
        # logp_pi = normal.log_prob(pi).sum(dim=1, keepdim=True)

        # # bounds
        # mu = torch.tanh(mu)
        # pi = torch.tanh(pi)
        # logp_pi -= torch.sum(torch.log(torch.clamp(1 - pi.pow(2), min=0, max=1) + EPS), dim=1, keepdim=True)

        tanh_normal = TanhNormal(mu, std)
        pi, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        logp_pi = tanh_normal.log_prob(pi, pre_tanh_value=pre_tanh_value)
        logp_pi = logp_pi.sum(dim=1, keepdim=True)

        return pi, logp_pi, mu, log_std


class SAC(nn.Module):
    def __init__(self, d_state, d_action, 
                 replay_size, batch_size, 
                 action_space_shape,
                 n_updates, n_hidden, 
                 gamma, alpha, reward_scale, lr, tau,
                 automatic_entropy_tuning=False):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.replay = Replay(d_state=d_state, d_action=d_action, size=replay_size)
        self.batch_size = batch_size

        self.n_updates = n_updates

        self.qf = ActionValueFunction_rlkit(self.d_state, d_action, n_hidden)
        self.qf_optim = Adam(self.qf.parameters(), lr=lr)

        self.vf = StateValueFunction_rlkit(self.d_state, n_hidden)
        self.vf_target = StateValueFunction_rlkit(self.d_state, n_hidden)
        self.vf_optim = Adam(self.vf.parameters(), lr=lr)
        for target_param, param in zip(self.vf_target.parameters(), self.vf.parameters()):
            target_param.data.copy_(param.data)

        self.policy = TanhGaussianPolicy(self.d_state, d_action, n_hidden)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.grad_clip = 5
        self.normalizer = None

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_space_shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.reward_scale = 1.

    @property
    def device(self):
        return next(self.parameters()).device

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.replay.setup_normalizer(normalizer)

    def __call__(self, states, eval=False):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        pi, _, mu, _ = self.policy(states)
        return mu if eval else pi

    def get_state_value(self, state):
        if self.normalizer is not None:
            state = self.normalizer.normalize_states(state)
        return self.vf(state)

    def reset_replay(self):
        self.replay.clear()

    def update(self, sample=None):
        global i
        if sample is None:
            sample = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, masks = [s.to(self.device) for s in sample]

        q1, q2 = self.qf(states, actions) # line 115, 116
        pi, logp_pi, mu, log_std = self.policy(states) # line 119
        q1_pi, q2_pi = self.qf(states, pi) # 152, 153
        v_pred = self.vf(states) # 117

        # alpha loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # target value network
        v_target = self.vf_target(next_states) # 143

        # min double-Q:
        min_q_pi = torch.min(q1_pi, q2_pi) # 151

        # targets for Q and V regression
        q_target = self.reward_scale * rewards + self.gamma * masks * v_target # 144 masks ?= (1-terminals)
        v_backup = min_q_pi - alpha * logp_pi # 155

        # policy losses
        pi_loss = torch.mean(alpha * logp_pi - min_q_pi) # 179
        pi_loss += 0.001 * mu.pow(2).mean()
        pi_loss += 0.001 * log_std.pow(2).mean()
        # pre_activation_weight=0 so disregard that

        # QF Loss
        #q1_loss = 0.5 * F.mse_loss(q1, q_target.detach())
        #q2_loss = 0.5 * F.mse_loss(q2, q_target.detach())
        q1_loss = F.mse_loss(q1, q_target.detach()) # 145
        q2_loss = F.mse_loss(q2, q_target.detach()) # 146

        # VF Loss
        # v_loss = 0.5 * F.mse_loss(v_pred, v_backup.detach())
        v_loss = F.mse_loss(v_pred, v_backup.detach()) # 156
        q_loss = q1_loss + q2_loss
        # value_loss = q1_loss + q2_loss + v_loss

        self.policy_optim.zero_grad()
        pi_loss.backward()
        #torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clip)
        self.policy_optim.step()

        self.qf_optim.zero_grad()
        q_loss.backward()
        #torch.nn.utils.clip_grad_value_(self.qf.parameters(), self.grad_clip)
        self.qf_optim.step()

        self.vf_optim.zero_grad()
        v_loss.backward()
        # value_loss.backward()
        #torch.nn.utils.clip_grad_value_(self.vf.parameters(), self.grad_clip)
        self.vf_optim.step()

        for target_param, param in zip(self.vf_target.parameters(), self.vf.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return v_loss.item(), q1_loss.item(), q2_loss.item(), pi_loss.item()

    def episode(self, env, warm_up=False, train=True, verbosity=0, _log=None):
        ep_returns = 0
        ep_length = 0
        states = env.reset()
        done = False
        while not done:
            if warm_up:
                actions = env.action_space.sample()
                actions = torch.from_numpy(actions)
            else:
                with torch.no_grad():
                    actions = self(states)

            next_states, rewards, done, _ = env.step(actions)
            self.replay.add(states, actions, rewards, next_states)
            if verbosity >= 3 and _log is not None:
                _log.info(
                    f'step_reward. mean: {torch.mean(rewards).item():5.2f} +- {torch.std(rewards).item():.2f} \
                        [{torch.min(rewards).item():5.2f}, {torch.max(rewards).item():5.2f}]')

            ep_returns += torch.mean(rewards).item()
            ep_length += 1

            states = next_states

        if train:
            if not warm_up:
                for _ in range(self.n_updates * ep_length):
                    self.update()

        return ep_returns
