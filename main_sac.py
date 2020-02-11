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

from sac_working import SAC
from sac_exploit import SAC as SAC_EX

import gym
import envs
from wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv, TorchEnv, NormalizedBoxEnv

from sacred import Experiment

from logger import get_logger

from torch.utils.tensorboard import SummaryWriter

ex = Experiment()
ex.logger = get_logger('max')
log_dir = 'runs/mujoco_cheetah/just_sac_old_gaussian1'
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
    # seed = 379385293

# noinspection PyUnusedLocal
@ex.config
def env_config():
    #env_name = 'MagellanHalfCheetah-v2'             # environment out of the defined magellan environments with `Magellan` prefix
    env_name = 'HalfCheetah-v2'             # environment out of the defined magellan environments with `Magellan` prefix
    n_eval_episodes = 3                             # number of episodes evaluated for each task
    env_noise_stdev = 0                             # standard deviation of noise added to state

    n_warm_up_steps = 256                          # number of steps to populate the initial buffer, actions selected randomly
    n_exploration_steps = 275#10001                     # total number of steps (including warm up) of exploration
    n_train_steps = 1000000
    env_horizon = 1000
    eval_freq = 200000                                # interval in steps for evaluating models on tasks in the environment
    data_buffer_size = n_exploration_steps + 1      # size of the data buffer (FIFO queue)
    buffer_load_file = None

    # misc.
    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]        # dimensionality of state
    d_action = env.action_space.shape[0]            # dimensionality of action
    del env


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
    else:
        device = torch.device('cpu')

    self_dir = os.path.dirname(sys.argv[0])
    dump_dir = os.path.join(self_dir,
                            'logs',
                            f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')

    os.makedirs(dump_dir, exist_ok=True)


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
    exploring_model_epochs = 50                    # number of training epochs in each training phase during exploration
    evaluation_model_epochs = 200                   # number of training epochs for evaluating the tasks
    batch_size = 256                                # batch size for training models
    learning_rate = 2e-4                            # learning rate for training models
    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance
    weight_decay = 1e-5                                # L2 weight decay on model parameters (good: 1e-5, default: 0)
    training_noise_stdev = 0                        # standard deviation of training noise applied on states, actions, next states
    grad_clip = 5                                   # gradient clipping to train model


# noinspection PyUnusedLocal
@ex.config
def policy_config():
    # common to both exploration and exploitation
    policy_actors = 128                             # number of parallel actors in imagination MDP
    policy_warm_up_episodes = 3                     # number of episodes with random actions before SAC on-policy data is collected (as a part of init)

    policy_replay_size = int(1e7)                   # SAC replay size
    policy_batch_size = 4096                        # SAC training batch size
    policy_reactive_updates = 100                   # number of SAC off-policy updates of `batch_size`
    policy_active_updates = 1                       # number of SAC on-policy updates per step in the imagination/environment

    policy_n_hidden = 256                           # policy hidden size (2 layers)
    policy_lr = 1e-3                                # SAC learning rate
    policy_gamma = 0.99                             # discount factor for SAC
    policy_tau = 0.005                              # soft target network update mixing factor

    buffer_reuse = True                             # transfer the main exploration buffer as off-policy samples to SAC
    use_best_policy = False                         # execute the best policy or the last one

    # exploration
    policy_explore_horizon = 50                     # length of sampled trajectories (planning horizon)
    policy_explore_episodes = 50                    # number of iterations of SAC before each episode
    policy_explore_alpha = 0.02                     # entropy scaling factor in SAC for exploration (utility maximisation)

    # exploitation
    policy_exploit_horizon = 100                    # length of sampled trajectories (planning horizon)
    policy_exploit_episodes = 250                   # number of iterations of SAC before each episode
    policy_exploit_alpha = 0.4                      # entropy scaling factor in SAC for exploitation (task return maximisation)


# noinspection PyUnusedLocal
@ex.config
def exploration():
    exploration_mode = 'active'                     # active or reactive

    model_train_freq = 25                           # interval in steps for training models. if `np.inf`, models are trained after every episode

    # utility_measure = 'renyi_div'                 # measure for calculating exploration utility of a particular (state, action)
    # utility_measure = 'traj_stdev'
    # utility_measure = 'cp_stdev'
    utility_measure =  'var'               
    renyi_decay = 0.1                               # decay to be used in calculating Renyi entropy

    utility_action_norm_penalty = 0                 # regularize to actions even when exploring
    action_noise_stdev = 0                          # noise added to actions


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


@ex.capture
def get_env(env_name, record, env_noise_stdev, mode='explore'):
    env = gym.make(env_name)
    #env = NormalizedBoxEnv(env)
    env = BoundedActionsEnv(env)

    #if env_noise_stdev:
    #    env = NoisyEnv(env, stdev=env_noise_stdev)
    #if mode == 'test':
    #    env = TorchEnv(env)
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
def train_epoch(model, buffer, optimizer, n_layers, batch_size, training_noise_stdev, grad_clip):
    losses = []
    for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(batch_size=batch_size):
 
        gen_loss = model.loss(tr_states, tr_actions, tr_state_deltas, training_noise_stdev=training_noise_stdev)
        
        for i, m in enumerate(optimizer):
            m.zero_grad()
        
        if gen_loss[1] is not None:
            loss, particle_values, svgd = gen_loss
            autograd.backward(particle_values, grad_tensors=svgd.detach())
            loss = loss.mean()
        else:
            loss = gen_loss[0]
            loss.backward()

        losses.append(loss.item())
        
        torch.nn.utils.clip_grad_value_(model.hypergan.generator.W1.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(model.hypergan.generator.W2.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(model.hypergan.generator.W3.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(model.hypergan.generator.W4.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(model.hypergan.generator.W5.parameters(), grad_clip)

        for i, m in enumerate(optimizer):
            m.step()
    return np.mean(losses)


@ex.capture
def fit_model(buffer, n_layers, n_epochs, step_num, verbosity, mode, _log, _run):
    model = get_model()
    model.setup_normalizer(buffer.normalizer)
    
    optimizer = [get_optimizer_factory()(model.hypergan.generator.W1.parameters()),
                  get_optimizer_factory()(model.hypergan.generator.W2.parameters()),
                  get_optimizer_factory()(model.hypergan.generator.W3.parameters()),
                  get_optimizer_factory()(model.hypergan.generator.W4.parameters()),
                  get_optimizer_factory()(model.hypergan.generator.W5.parameters())]

    if verbosity:
        _log.info(f"step: {step_num}\t training")

    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        if verbosity >= 2:
            _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')

    _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")

    if mode == 'explore':
        writer.add_scalar("explore_loss", tr_loss, step_num)
    elif mode == 'exploit':
        writer.add_scalar("exploit_loss", tr_loss, step_num)

    return model


"""
Planning
"""


@ex.capture
def get_policy(buffer, model, measure, mode,
               d_state, d_action, policy_replay_size, policy_batch_size, policy_active_updates,
               policy_n_hidden, policy_lr, policy_gamma, policy_tau, policy_explore_alpha, policy_exploit_alpha, buffer_reuse,
               device, verbosity, _log):

    if verbosity:
        _log.info("... getting fresh agent")

    policy_alpha = policy_explore_alpha if mode == 'explore' else policy_exploit_alpha
    #if mode == 'sac':
    #agent = SAC_EX(d_state=d_state, d_action=d_action, replay_size=policy_replay_size, 
    #            n_updates=policy_active_updates, n_hidden=policy_n_hidden)
    #else:
    agent = SAC(d_state=d_state, d_action=d_action, replay_size=policy_replay_size, batch_size=policy_batch_size,
                n_updates=policy_active_updates, n_hidden=policy_n_hidden, gamma=policy_gamma, alpha=policy_alpha,
                lr=policy_lr, tau=policy_tau)

    agent = agent.to(device)
    # if mode != 'sac':
    # agent.setup_normalizer(model.normalizer)

    if not buffer_reuse:
        return agent

    if verbosity:
        _log.info("... transferring exploration buffer")

    size = len(buffer)
    for i in range(0, size, 1024):
        j = min(i + 1024, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        s, a, ns = s.to(device), a.to(device), ns.to(device)
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
    action = actions[0].detach().data.cpu().numpy()
    policy_value = torch.mean(agent.get_state_value(current_state)).item()
    return action, mdp, agent, policy_value


@ex.capture
def evaluate_agent(agent, step_num):
    env = get_env()
    env = TorchEnv(env)
    ep_returns = []
    ep_return = 0
    state = env.reset()
    done = False
    while not done:
        action = agent(state, eval=True)
        next_state, reward, done, _ = env.step(action)
        ep_return += reward.squeeze().detach().data.cpu().numpy()
        state = next_state
    return ep_return


@ex.capture
def train(env, agent, n_train_steps, verbosity, env_horizon, _run, _log): 
    agent.reset_replay()
    ep_returns = []
    n_episodes = int(n_train_steps / env_horizon)
    for ep_i in range(n_episodes):
        ep_return = agent.episode(env=TorchEnv(env), verbosity=verbosity, _log=_log)
        ep_returns.append(ep_return)

        step_return = ep_return / env_horizon
        _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")
        writer.add_scalar("step_return", step_return, ep_i)
        
        task_step_num = 1000 * ep_i
        avg_return = evaluate_agent(agent, task_step_num)
        _log.info(f"task_step: {task_step_num}, evaluate:\taverage_return = {np.round(avg_return, 4)}")
        writer.add_scalar(f"evaluate_return", avg_return, task_step_num)

    return max(ep_returns)

@ex.capture
def imagined_train(state, buffer, model, measure, policy_actors, policy_reactive_updates,
        policy_warm_up_episodes, policy_explore_horizon, policy_explore_episodes, 
        verbosity, _run, _log):

    mdp = Imagination(horizon=policy_explore_horizon, 
            n_actors=policy_actors, model=model, measure=measure)
    agent = get_policy(buffer=buffer, model=model, measure=measure, mode='sac')#, buffer_reuse=True)

    mdp.update_init_state(state)
    
    # reactive updates
    for update_idx in range(policy_reactive_updates):
        agent.update()
        
    # active training
    agent.reset_replay()
    ep_returns = []
    for ep_i in range(policy_explore_episodes):
        warm_up = True if (ep_i < policy_warm_up_episodes) else False
        ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
        ep_returns.append(ep_return)

        if verbosity:
            step_return = ep_return / policy_explore_horizon
            _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

    return agent
    

@ex.capture
def act(state, agent, mdp, buffer, model, measure, mode, exploration_mode,
        policy_actors, policy_warm_up_episodes, use_best_policy, policy_reactive_updates,
        policy_explore_horizon, policy_exploit_horizon,
        policy_explore_episodes, policy_exploit_episodes,
        verbosity, _run, _log):

    if mode == 'explore':
        policy_horizon = policy_explore_horizon
        policy_episodes = policy_explore_episodes
    elif mode == 'exploit':
        policy_horizon = policy_exploit_horizon
        policy_episodes = policy_exploit_episodes
    else:
        raise Exception("invalid acting mode")

    fresh_agent = True if agent is None else False
    if mdp is None:
        #print ('creating imaginary mdp')
        mdp = Imagination(horizon=policy_horizon, n_actors=policy_actors, model=model, measure=measure)

    if fresh_agent:
        #rint ('getting new agent')
        agent = get_policy(buffer=buffer, model=model, measure=measure, mode=mode)

    # update state to current env state
    mdp.update_init_state(state)

    if not fresh_agent:
        # agent is not stale, use it to return action
        #print ('returning action -- old agent')
        return get_action(mdp, agent)

    # reactive updates
    for update_idx in range(policy_reactive_updates):
        agent.update()

    # active updates
    perform_active_exploration = (mode == 'explore' and exploration_mode == 'active')
    perform_exploitation = (mode == 'exploit')
    if perform_active_exploration or perform_exploitation:

        # to be fair to reactive methods, clear real env data in SAC buffer, to prevent further gradient updates from it.
        # for active exploration, only effect of on-policy training remains
        if perform_active_exploration:
            #print ('resetting replay buffer -- on policy training')
            agent.reset_replay()

        ep_returns = []
        best_return, best_params = -np.inf, deepcopy(agent.state_dict())
        for ep_i in range(policy_episodes):
            warm_up = True if ((ep_i < policy_warm_up_episodes) and fresh_agent) else False
            ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
            ep_returns.append(ep_return)

            if use_best_policy and ep_return > best_return:
                best_return, best_params = ep_return, deepcopy(agent.state_dict())

            if verbosity:
                step_return = ep_return / policy_horizon
                _log.info(f"\tep: {ep_i}\taverage step return: {np.round(step_return, 3)}")

        if use_best_policy:
            agent.load_state_dict(best_params)

        if mode == 'explore' and len(ep_returns) >= 3:
            first_return = ep_returns[0]
            last_return = max(ep_returns) if use_best_policy else ep_returns[-1]
            """
            writer.add_scalar("policy_improvement_first_return", first_return / policy_horizon)
            writer.add_scalar("policy_improvement_second_return", ep_returns[1] / policy_horizon)
            writer.add_scalar("policy_improvement_last_return", last_return / policy_horizon)
            writer.add_scalar("policy_improvement_max_return", max(ep_returns) / policy_horizon)
            writer.add_scalar("policy_improvement_min_return", min(ep_returns) / policy_horizon)
            writer.add_scalar("policy_improvement_median_return", np.median(ep_returns) / policy_horizon)
            writer.add_scalar("policy_improvement_first_last_delta", (last_return - first_return) / policy_horizon)
            writer.add_scalar("policy_improvement_second_last_delta", (last_return - ep_returns[1]) / policy_horizon)
            writer.add_scalar("policy_improvement_median_last_delta", (last_return - np.median(ep_returns)) / policy_horizon)
            """
    return get_action(mdp, agent)


"""
Evaluation and Check-pointing
"""


@ex.capture
def transition_novelty(state, action, next_state, model, renyi_decay):
    state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
    action = torch.from_numpy(action).float().unsqueeze(0).to(model.device)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(model.device)

    with torch.no_grad():
        mu, var = model.forward_all(state, action)
    #measure = JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay)
    measure = SimpleVarianceUtility()
    #measure = CompoundProbabilityStdevUtilityMeasure()
    v = measure(state, action, next_state, mu, var, model)
    return v.item()


@ex.capture
def checkpoint(buffer, step_num, dump_dir, _run):
    buffer_file = f'{dump_dir}/{step_num}.buffer'
    #with gzip.open(buffer_file, 'wb') as f:
    #    pickle.dump(buffer, f)
    _run.add_artifact(buffer_file)


"""
Main Functions
"""
@ex.capture
def load_checkpoint(buffer_load_file, _run):
    print ("loading from checkpoint: ", buffer_load_file)
    with gzip.open(buffer_load_file, 'rb') as f:
        buffer = pickle.load(f, encoding='latin1')  # load from buffer file
    step_num = int(buffer_load_file.split('.')[0].split('/', 10)[-1])
    return buffer, step_num



@ex.capture
def do_max_exploration(seed, buffer_load_file, action_noise_stdev, n_exploration_steps, n_warm_up_steps,
                       model_train_freq, exploring_model_epochs,
                       eval_freq, checkpoint_frequency, render, record, dump_dir, _config, _log, _run):

    env = get_env()
    env.seed(seed)
    atexit.register(lambda: env.close())

    buffer = get_buffer()
    exploration_measure = get_utility_measure()

    if _config['normalize_data']:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    model = None
    s_num = 1
    if buffer_load_file is not None:
        buffer, s_num = load_checkpoint()
        model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=s_num, mode='explore')
 
    mdp = None
    agent = None
    average_performances = []

    state = env.reset()

    for step_num in range(s_num, n_exploration_steps + 1):
        policy_values = []
        action_norms = []
        if step_num > n_warm_up_steps:
            action, mdp, agent, policy_value = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=exploration_measure, mode='explore')

            writer.add_scalar("action_norm", np.sum(np.square(action)), step_num)
            writer.add_scalar("exploration_policy_value", policy_value, step_num)

            if action_noise_stdev:
                action = action + np.random.normal(scale=action_noise_stdev, size=action.shape)
        else:
            action = env.action_space.sample()

        # real env rollout
        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, next_state)

        if step_num > n_warm_up_steps:
            writer.add_scalar("experience_novelty", transition_novelty(state, action, next_state, model=model), step_num)

        if done:
            _log.info(f"step: {step_num}\tepisode complete")
            agent = None
            mdp = None

            if record:
                new_video_filename = f"{dump_dir}/exploration_{step_num}.mp4"
                next_state = env.reset(filename=new_video_filename)
                _run.add_artifact(video_filename)
                video_filename = new_video_filename
            else:
                next_state = env.reset()

        state = next_state

        if step_num < n_warm_up_steps:
            continue

        episode_done = done
        train_at_end_of_episode = (model_train_freq is np.inf)
        time_to_update = ((step_num % model_train_freq) == 0)
        just_finished_warm_up = (step_num == n_warm_up_steps)
        if (train_at_end_of_episode and episode_done) or time_to_update or just_finished_warm_up:
            model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=step_num, mode='explore')

            # discard old solution and MDP as models changed
            mdp = None
            agent = None

        time_to_checkpoint = ((step_num % checkpoint_frequency) == 0)
        if time_to_checkpoint:
            checkpoint(buffer=buffer, step_num=step_num)

    _log.info(f"intrinsic training finished")
    #if agent is None:
    #    agent = imagined_train(state=state, buffer=buffer, model=model, measure=exploration_measure)

    agent = get_policy(buffer=None, model=None, measure=None, buffer_reuse=False,
                       policy_batch_size=4096, policy_lr=1e-3, mode='explore')
    _log.info(f"starting extrinsic training")
    max_return = train(env=env, agent=agent) 

    return max_return


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

