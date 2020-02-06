#!/usr/bin/env python

import numpy as np

import torch

import os
import sys
from datetime import datetime
import atexit
import gzip
import pickle
from copy import deepcopy
import torch.autograd as autograd

from buffer import Buffer
from models import Model
from utilities import CompoundProbabilityStdevUtilityMeasure, JensenRenyiDivergenceUtilityMeasure, \
    TrajectoryStdevUtilityMeasure, PredictionErrorUtilityMeasure, SimpleVarianceUtility
from normalizer import TransitionNormalizer
from imagination import Imagination

from sac import SAC
import gym
import envs
from wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv, TorchEnv, NormalizedBoxEnv

from sacred import Experiment

from logger import get_logger

from torch.utils.tensorboard import SummaryWriter

ex = Experiment()
ex.logger = get_logger('max')
log_dir = 'runs/cheetah/run_v2'
writer = SummaryWriter(log_dir=log_dir)
print ('writing to', log_dir)

k = 0
l = 0
# noinspection PyUnusedLocal
@ex.config
def config():
    max_exploration = False
    random_exploration = False
    exploitation = False
    ant_coverage = False
    rotation_coverage = False


# noinspection PyUnusedLocal
@ex.config
def infra_config():
    verbosity = 0                                   # level of logging/printing on screen
    render = False                                  # render the environment visually (warning: could open too many windows)
    record = False                                  # record videos of episodes (warning: could be slower and use up disk space)
    save_eval_agents = False                        # save evaluation agent (sac module objects)

    checkpoint_frequency = 2000                     # dump buffer with normalizer every checkpoint_frequency steps

    disable_cuda = False                            # if true: do not ues cuda even though its available
    omp_num_threads = 1                           # for high CPU count machines

    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    self_dir = os.path.dirname(sys.argv[0])
    dump_dir = os.path.join(self_dir,
                            'logs',
                            f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')

    os.makedirs(dump_dir, exist_ok=True)


# noinspection PyUnusedLocal
@ex.config
def env_config():
    env_name = 'HalfCheetah-v2'             # environment out of the defined magellan environments with `Magellan` prefix
    env_exploit_name = 'MagellanHalfCheetah-v2'     # environment out of the defined magellan environments with `Magellan` prefix
    env_noise_stdev = 0                             # standard deviation of noise added to state
    n_warm_up_steps = 256                          # number of steps to populate the initial buffer, actions selected randomly
    n_exploration_steps = 10000                     # total number of steps (including warm up) of exploration
    n_task_steps = 990000
    env_horizon = 1000
    data_buffer_size = int(1e+6) + 1      # size of the data buffer (FIFO queue)
    action_noise_stdev = 0                          # noise added to actions

    buffer_load_file = None                         # exact path to load a buffer (checkpoint)

    # misc.
    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]        # dimensionality of state
    d_action = env.action_space.shape[0]            # dimensionality of action
    del env


# noinspection PyUnusedLocal
@ex.config
def model_arch_config():
    ensemble_size = 32                              # number of models in the bootstrap ensemble
    n_hidden = 512                                  # number of hidden units in each hidden layer (hidden layer size)
    n_layers = 4                                    # number of hidden layers in the model (at least 2)
    non_linearity = 'swish'                         # activation function: can be 'leaky_relu' or 'swish'


# noinspection PyUnusedLocal
@ex.config
def model_training_config():
    model_train_freq = 25                           # interval in steps for training models. `np.inf`: train after every episode
    exploring_model_epochs = 100                    # number of training epochs in each training phase during exploration
    evaluation_model_epochs = 200                   # number of training epochs for evaluating the tasks
    batch_size = 256                                # batch size for training models
    learning_rate = 1e-3                            # learning rate for training models
    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance
    weight_decay = 0                                # L2 weight decay on model parameters (good: 1e-5, default: 0)
    training_noise_stdev = 0                        # standard deviation of training noise applied on states, actions, next states
    grad_clip = 5                                   # gradient clipping to train model


# noinspection PyUnusedLocal
@ex.config
def policy_config():
    """ common parameters for exploration and task """
    policy_actors = 128                             # number of parallel actors in imagination MDP
    policy_warm_up_episodes = 3                     # number of episodes with random actions before SAC on-policy training 
    policy_n_hidden = 256                           # policy hidden size (2 layers)
    policy_replay_size = int(1e6) + 1                   # SAC replay size
    policy_gamma = 0.99                             # discount factor for SAC
    policy_tau = 0.005                              # soft target network update mixing factor
    policy_reward_scale = 5
    policy_eval_freq = 500                                # interval in steps for evaluating models on tasks in the environment
    n_policy_eval_episodes = 3                             # number of episodes evaluated for each task

    """ task agent parameters """
    policy_task_batch_size = 256
    policy_task_lr = 3e-4
    policy_task_alpha = 1.0
    buffer_reuse_task = True                             # transfer the main exploration buffer as off-policy samples to SAC

    policy_task_active_updates = 1
    policy_task_train_freq = 1

    """ explore agent parameters """
    policy_explore_batch_size = 4096                        # SAC training batch size
    policy_explore_lr = 1e-3                                # SAC learning rate
    policy_explore_alpha = 0.1                     # entropy scaling factor in SAC for exploration (utility maximisation)
    buffer_reuse_explore = True                             # transfer the main exploration buffer as off-policy samples to SAC

    policy_explore_active_updates = 1               # number of SAC on-policy updates per step in the imagination/environment
    policy_explore_reactive_updates = 100                   # number of SAC off-policy updates of `batch_size`
    policy_explore_horizon = 50                     # length of sampled trajectories (planning horizon)
    policy_explore_episodes = 50                    # number of iterations of SAC before each episode
    use_best_policy = False                         # execute the best policy or the last one


# noinspection PyUnusedLocal
@ex.config
def measure_config():
    utility_measure = 'renyi_div'                 # measure for calculating exploration utility of a particular (state, action)
    # utility_measure = 'traj_stdev'
    # utility_measure = 'cp_stdev'
    # utility_measure =  'var'               
    renyi_decay = 0.1                               # decay to be used in calculating Renyi entropy
    utility_action_norm_penalty = 0                 # regularize to actions even when exploring


# noinspection PyUnusedLocal
@ex.named_config
def max_explore():
    max_exploration = True


# noinspection PyUnusedLocal
@ex.named_config
def random_explore():
    random_exploration = True


# noinspection PyUnusedLocal
@ex.named_config
def exploit():
    exploitation = True
    buffer_file = ''
    benchmark_utility = False


"""
Initialization Helpers
"""

""" Either returns the env or the exploit env,
Mixing Envs is hard since Magellan has two more states than HalfCheetah """
@ex.capture
def get_env(env_name): #, env_exploit_name, record, env_noise_stdev, mode='explore'):
    env = gym.make(env_name)
    env = NormalizedBoxEnv(env)
    env = TorchEnv(env)
    return env


@ex.capture
def get_model(d_state, d_action, ensemble_size, n_hidden, n_layers,
              non_linearity, device):

    model = Model(d_action=d_action,
                  d_state=d_state,
                  ensemble_size=ensemble_size,
                  n_hidden=n_hidden,
                  n_layers=n_layers,
                  non_linearity=non_linearity,
                  device=device)
    return model


@ex.capture
def get_buffer(d_state, d_action, ensemble_size, data_buffer_size):
    return Buffer(d_action=d_action,
                  d_state=d_state,
                  ensemble_size=ensemble_size,
                  buffer_size=data_buffer_size)


@ex.capture
def get_optimizer_factory(learning_rate, weight_decay):
    return lambda params: torch.optim.Adam(params,
                                           lr=learning_rate,
                                           weight_decay=weight_decay)


@ex.capture
def get_utility_measure(utility_measure, utility_action_norm_penalty, renyi_decay):
    if utility_measure == 'cp_stdev':
        return CompoundProbabilityStdevUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'renyi_div':
        return JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay, action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'traj_stdev':
        return TrajectoryStdevUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'pred_err':
        return PredictionErrorUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    elif utility_measure == 'var':
        return SimpleVarianceUtility(action_norm_penalty=utility_action_norm_penalty)
    else:
        raise Exception('invalid utility measure')


"""
Model Training
"""

@ex.capture
def train_epoch(model, buffer, optimizer, batch_size, training_noise_stdev, grad_clip):
    losses = []
    for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(batch_size=batch_size):
        optimizer.zero_grad()
        loss = model.loss(tr_states, tr_actions, tr_state_deltas, training_noise_stdev=training_noise_stdev)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


@ex.capture
def fit_model(buffer, n_epochs, step_num, verbosity, _log, _run):
    model = get_model()
    model.setup_normalizer(buffer.normalizer)
    optimizer = get_optimizer_factory()(model.parameters())

    if verbosity:
        _log.info(f"step: {step_num}\t training")

    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        if verbosity >= 2:
            _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

    _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")

    _run.log_scalar("explore_loss", tr_loss, step_num)

    return model


"""
Planning
"""

@ex.capture
def get_policy(buffer, model, env, measure, mode, d_state, d_action, device, verbosity, _log,
               policy_n_hidden, policy_replay_size, policy_gamma, policy_tau, policy_reward_scale, # common params
               policy_task_batch_size, policy_task_lr, policy_task_alpha, # task params
               policy_explore_batch_size, policy_explore_lr, policy_explore_alpha, # explore params
               policy_explore_active_updates, buffer_reuse_explore): 

    if verbosity:
        _log.info("... getting fresh agent")

    if mode == 'explore':
        policy_batch_size = policy_explore_batch_size
        policy_alpha = policy_explore_alpha 
        policy_lr = policy_explore_lr
    else:
        policy_batch_size = policy_task_batch_size
        policy_alpha = policy_task_alpha
        policy_lr = policy_task_lr

    agent = SAC(d_state=d_state, d_action=d_action, replay_size=policy_replay_size,  
                n_hidden=policy_n_hidden, n_updates=policy_explore_active_updates,
                gamma=policy_gamma, tau=policy_tau, reward_scale=policy_reward_scale,
                batch_size=policy_batch_size, alpha=policy_alpha, lr=policy_lr,
                action_space_shape=env.action_space.shape)

    agent = agent.to(device)
    # agent.setup_normalizer(model.normalizer)

    if not (buffer_reuse_explore and mode == 'explore'):
        return agent

    if verbosity:
        _log.info("... transferring exploration buffer")

    size = len(buffer)
    for i in range(0, size, 1000):
        j = min(i + 1000, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        # s, a, ns = s.to(device), a.to(device), ns.to(device)
        with torch.no_grad():
            mu, var = model.forward_all(s, a)
        r = measure(s, a, ns, mu, var, model)
        agent.replay.add(s, a, r, ns)

    if verbosity:
        _log.info("... transferred exploration buffer")

    return agent


def get_action(mdp, agent):
    current_state = mdp.reset()
    actions = agent(current_state, eval=True)
    action = actions[0].detach() #.data.cpu().numpy()
    # policy_value = torch.mean(agent.get_state_value(current_state)).item()
    return action, mdp, agent #, policy_value


@ex.capture
def transfer_buffer_to_agent(buffer, agent, device, verbosity, _log):
    size = len(buffer)
    for i in range(0, size, 1000):
        j = min(i + 1000, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        # s, a, ns = s.to(device), a.to(device), ns.to(device)
        r = buffer.rewards[i:j] #.to(device)
        agent.replay.add(s, a, r, ns)

    _log.info("... transferred exploration buffer to task agent")

    return agent

@ex.capture
def act(state, agent, mdp, buffer, model, measure,
        policy_actors, policy_warm_up_episodes, use_best_policy, 
        policy_explore_reactive_updates, policy_explore_horizon,
        policy_explore_episodes, verbosity, _run, _log):

    fresh_agent = True if agent is None else False
    if mdp is None:
        #print ('creating imaginary mdp')
        mdp = Imagination(horizon=policy_explore_horizon, n_actors=policy_actors, model=model, measure=measure)

    if fresh_agent:
        #rint ('getting new agent')
        agent = get_policy(buffer=buffer, model=model, env=mdp, measure=measure, mode='explore')

    # update state to current env state
    mdp.update_init_state(state)

    if not fresh_agent:
        return get_action(mdp, agent)

    # reactive updates
    for update_idx in range(policy_explore_reactive_updates):
        agent.update()

    # active updates
    agent.reset_replay()
    for ep_i in range(policy_explore_episodes):
        warm_up = True if (ep_i < policy_warm_up_episodes) else False
        _ = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
    return get_action(mdp, agent)


"""
Evaluation and Check-pointing
"""

@ex.capture
def evaluate_agent(agent, step_num, n_policy_eval_episodes):
    env = get_env()
    ep_returns = []
    for ep_idx in range(n_policy_eval_episodes):
        ep_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            ep_return += reward.squeeze().detach().data.cpu().numpy()
            state = next_state
        ep_returns.append(ep_return)
    return np.mean(ep_returns)


@ex.capture
def checkpoint(buffer, step_num, dump_dir, _run):
    buffer_file = f'{dump_dir}/{step_num}.buffer'
    with gzip.open(buffer_file, 'wb') as f:
        pickle.dump(buffer, f)
    _run.add_artifact(buffer_file)


@ex.capture
def load_checkpoint(buffer_load_file, _run):
    print ("loading from checkpoint: ", buffer_load_file)
    with gzip.open(buffer_load_file, 'rb') as f:
        buffer = pickle.load(f, encoding='latin1')  # load from buffer file
    step_num = int(buffer_load_file.split('.')[0].split('/', 10)[-1])
    return buffer, step_num


"""
Main Functions
"""


@ex.capture
def do_max_exploration(seed, action_noise_stdev, buffer_load_file,
                       n_exploration_steps, n_task_steps, n_warm_up_steps, 
                       model_train_freq, exploring_model_epochs, policy_eval_freq,
                       policy_task_train_freq, policy_task_active_updates, 
                       policy_task_batch_size, policy_task_lr, policy_task_alpha,
                       buffer_reuse_task, checkpoint_frequency,
                       device, dump_dir, _config, _log, _run):

    env = get_env()
    env.seed(seed)
    atexit.register(lambda: env.close())

    buffer = get_buffer()
    exploration_measure = get_utility_measure()

    if _config['normalize_data']:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    model = None

    step_num = 1
    if buffer_load_file is not None:
        buffer, step_num = load_checkpoint()
        model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=step_num)

    mdp = None
    agent = None

    """ intrinsic training stage """
    state = env.reset()
    for explore_step_num in range(step_num, n_exploration_steps + 1):
        # policy_values = []
        # action_norms = []
        if explore_step_num > n_warm_up_steps:
            action, mdp, agent = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=exploration_measure)
            # writer.add_scalar("action_norm", np.sum(np.square(action)), explore_step_num)
            # writer.add_scalar("exploration_policy_value", policy_value, step_num)

            if action_noise_stdev:
                action = action + action_noise_stdev * torch.randn(action.shape)
        else:
            action = env.action_space.sample()
            action = torch.from_numpy(action).float().to(device)
            action = action.squeeze(0)

        # real env rollout
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state)

        if done:
            _log.info(f"step: {explore_step_num}\tepisode complete")
            # agent = None
            # mdp = None
            next_state = env.reset()

        state = next_state

        if explore_step_num < n_warm_up_steps:
            continue

        time_to_update = ((explore_step_num % model_train_freq) == 0)
        just_finished_warm_up = (explore_step_num == n_warm_up_steps)
        if time_to_update or just_finished_warm_up:
            model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=explore_step_num)
            # discard old solution and MDP as models changed
            mdp = None
            agent = None

        time_to_checkpoint = ((explore_step_num % checkpoint_frequency) == 0)
        if time_to_checkpoint:
            checkpoint(buffer=buffer, step_num=explore_step_num)

    _log.info(f"intrinsic training finished")


    """ extrinsic training stage """
    agent = None
    agent = get_policy(buffer=buffer, model=model, env=env, measure=exploration_measure, mode='task')
    agent = transfer_buffer_to_agent(buffer, agent)

    _log.info(f"starting extrinsic training")
    ep_returns = []
    for task_step_num in range(1, n_task_steps + 1):
        with torch.no_grad():
            action = agent(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay.add(state, action, reward, next_state)

        if done:
            _log.info(f"task_step: {task_step_num}\tepisode complete")
            next_state = env.reset()
        state = next_state

        # train task policy
        if task_step_num % policy_task_train_freq == 0:
            for _ in range(policy_task_active_updates):
                agent.update()

        # evaluate task policy
        if task_step_num % policy_eval_freq == 0:
            avg_return = evaluate_agent(agent, task_step_num)
            _log.info(f"task_step: {task_step_num}, evaluate:\taverage_return = {np.round(avg_return, 4)}")
            writer.add_scalar(f"evaluate_return", avg_return, task_step_num)
            ep_returns.append(avg_return)

    return max(ep_returns)
    # return max(average_performances)


@ex.automain
def main(max_exploration, random_exploration, exploitation, seed, omp_num_threads):
    ex.commands["print_config"]()

    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if max_exploration:
        return do_max_exploration()

