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
from wrappers import BoundedActionsEnv, RecordedEnv, NoisyEnv, TorchEnv

from sacred import Experiment

from logger import get_logger

from torch.utils.tensorboard import SummaryWriter

ex = Experiment()
ex.logger = get_logger('max')
log_dir = 'runs/block/a1e-3_lr2e-4_32i16h_means_100_grad4_4'
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
def env_config():
    env_name = 'MagellanHalfCheetah-v2'             # environment out of the defined magellan environments with `Magellan` prefix
    env_base = env_name
    n_eval_episodes = 3                             # number of episodes evaluated for each task
    env_noise_stdev = 0                             # standard deviation of noise added to state

    n_warm_up_steps = 256                          # number of steps to populate the initial buffer, actions selected randomly
    n_exploration_steps = 1000000                    # total number of steps (including warm up) of exploration
    n_train_steps = 1000000
    env_horizon = 1000
    eval_freq = 500                                 # interval in steps for evaluating models on tasks in the environment
    data_buffer_size = n_exploration_steps + 1      # size of the data buffer (FIFO queue)

    # misc.
    env = gym.make(env_base)
    d_state = env.observation_space.shape[0]
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
    exploring_model_epochs = 100                    # number of training epochs in each training phase during exploration
    evaluation_model_epochs = 200                   # number of training epochs for evaluating the tasks
    batch_size = 256                                # batch size for training models
    learning_rate = 2e-4                            # learning rate for training models
    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance
    weight_decay = 1e-5                                # L2 weight decay on model parameters (good: 1e-5, default: 0)
    training_noise_stdev = 0                        # standard deviation of training noise applied on states, actions, next states
    grad_clip = 4                                   # gradient clipping to train model


# noinspection PyUnusedLocal
@ex.config
def policy_config():
    # common to both exploration and exploitation
    policy_actors = 128                             # number of parallel actors in imagination MDP
    policy_warm_up_episodes = 3                     # number of episodes with random actions before SAC on-policy data is collected (as a part of init)

    policy_replay_size = int(1e+6) + 1                   # SAC replay size
    policy_batch_size = 4096                        # SAC training batch size
    policy_reactive_updates = 100                   # number of SAC off-policy updates of `batch_size`
    # policy_initial_updates = 5000
    policy_active_updates = 1                       # number of SAC on-policy updates per step in the imagination/environment
    agent_train_freq = 1
    agent_batch_size = 256
    agent_active_updates = 1
    explore_agent_updates = 1000

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

    model_train_freq = 1000                           # interval in steps for training models. if `np.inf`, models are trained after every episode
    explore_rollout_freq = 100
    n_explore_rollout_steps = 20

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
def get_env(env_name, record, env_noise_stdev):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    if env_noise_stdev:
        env = NoisyEnv(env, stdev=env_noise_stdev)

    if record:
        env = RecordedEnv(env)

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

    agent = SAC(d_state=d_state, d_action=d_action, replay_size=policy_replay_size, batch_size=policy_batch_size,
                n_updates=policy_active_updates, n_hidden=policy_n_hidden, gamma=policy_gamma, alpha=policy_alpha,
                lr=policy_lr, tau=policy_tau)

    agent = agent.to(device)
    # if model is not None:
    #     agent.setup_normalizer(model.normalizer)

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


@ex.capture
def transfer_buffer_to_agent(buffer, agent, model, measure, device, verbosity):
    agent.reset_replay()
    size = len(buffer)
    for i in range(0, size, 1000):
        j = min(i + 1000, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        s, a, ns = s.to(device), a.to(device), ns.to(device)
        # r = buffer.rewards[i:j].to(device)
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
def train(env, agent, n_train_steps, verbosity, env_horizon, _run, _log): 
    # agent.reset_replay()
    ep_returns = []
    n_episodes = int(n_train_steps / env_horizon)
    for ep_i in range(n_episodes):
        # ep_return = agent.rewrad_episode(env=env, reward_func=reward_function, verbosity=verbosity, _log=_log)
        ep_return = agent.episode(env=TorchEnv(env), verbosity=verbosity, _log=_log)
        ep_returns.append(ep_return)
        _log.info(f"\tep: {ep_i}\taverage step return: {np.round(ep_return, 3)}")

    return max(ep_returns)

@ex.capture
def imagined_train(state, buffer, model, measure, policy_actors, policy_reactive_updates,
        policy_warm_up_episodes, policy_explore_horizon, policy_explore_episodes, 
        verbosity, _run, _log):

    mdp = Imagination(horizon=policy_explore_horizon, 
            n_actors=policy_actors, model=model, measure=measure)
    agent = get_policy(buffer=buffer, model=model, measure=measure, mode='explore')

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
            _log.info(f"\tep: {ep_i}\taverage step return: {np.round(ep_return, 3)}")

    return agent
    

@ex.capture
def act(state, agent, buffer, model, measure, mode, exploration_mode,
        policy_actors, policy_warm_up_episodes, use_best_policy,
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

    mdp = Imagination(horizon=policy_horizon, n_actors=policy_actors, model=model, measure=measure)
    mdp.update_init_state(state)

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
            warm_up = True if (ep_i < policy_warm_up_episodes) else False
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
    return agent


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
def evaluate_task(env, model, buffer, task, render, filename, record, save_eval_agents, verbosity, _run, _log):
    env.seed(np.random.randint(2 ** 32 - 1))

    video_filename = f'{filename}.mp4'
    if record:
        state = env.reset(filename=video_filename)
    else:
        state = env.reset()

    ep_return = 0
    agent = None
    mdp = None
    done = False
    novelty = []
    while not done:
        action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=task.measure, mode='exploit')
        next_state, _, done, info = env.step(action)

        n = transition_novelty(state, action, next_state, model=model)
        novelty.append(n)

        reward = task.reward_function(state, action, next_state)
        if verbosity >= 3:
            _log.info(f'reward: {reward:5.2f} trans_novelty: {n:5.2f} action: {action}')
        ep_return += reward

        if render:
            env.render()

        state = next_state

    env.close()

    if record:
        _run.add_artifact(video_filename)

    if save_eval_agents:
        agent_filename = f'{filename}_agent.pt'
        torch.save(agent.state_dict(), agent_filename)
        _run.add_artifact(agent_filename)

    return ep_return, np.mean(novelty)


@ex.capture
def evaluate_tasks(buffer, step_num, n_eval_episodes, evaluation_model_epochs, render, ant_coverage, rotation_coverage, dump_dir, _log, _run):
    # Uncomment for exploration coverage in ant
    if ant_coverage:
        from envs.ant import rate_buffer
        coverage = rate_buffer(buffer=buffer)
        writer.add_scalar("coverage", coverage, step_num)
        _run.result = coverage
        _log.info(f"coverage: {coverage}")
        return coverage

    if rotation_coverage:
        from envs.handblock import rotation_buffer
        coverage = rotation_buffer(buffer=buffer)
        writer.add_scalar("coverage", coverage, step_num)
        _run.result = coverage
        _log.info(f"coverage: {coverage}")
        return coverage

    model = fit_model(buffer=buffer, n_epochs=evaluation_model_epochs, step_num=step_num, mode='exploit')
    env = get_env()

    average_returns = []
    for task_name, task in env.unwrapped.tasks.items():
        task_returns = []
        task_novelty = []
        for ep_idx in range(1, n_eval_episodes + 1):
            filename = f"{dump_dir}/evaluation_{step_num}_{task_name}_{ep_idx}"
            ep_return, ep_novelty = evaluate_task(env=env, model=model, buffer=buffer, task=task, render=render, filename=filename)

            _log.info(f"task: {task_name}\tepisode: {ep_idx}\treward: {np.round(ep_return, 4)}")
            task_returns.append(ep_return)
            task_novelty.append(ep_novelty)

        average_returns.append(task_returns)
        _log.info(f"task: {task_name}\taverage return: {np.round(np.mean(task_returns), 4)}")
        writer.add_scalar(f"task_{task_name}_return", np.mean(task_returns), step_num)
        writer.add_scalar(f"task_{task_name}_episode_novelty", np.mean(task_novelty), step_num)

    average_return = np.mean(average_returns)
    writer.add_scalar("average_return", average_return, step_num)
    _run.result = average_return
    return average_return


@ex.capture
def evaluate_utility(buffer, exploring_model_epochs, model_train_freq, n_eval_episodes, _log, _run):
    env = get_env()
    env.seed(np.random.randint(2 ** 32 - 1))

    measure = get_utility_measure(utility_measure='var', utility_action_norm_penalty=0)

    achieved_utilities = []
    for ep_idx in range(1, n_eval_episodes + 1):
        state = env.reset()
        ep_utility = 0
        ep_length = 0

        model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=0, mode='explore')
        agent = None
        mdp = None
        done = False

        while not done:
            action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model, measure=measure, mode='explore')
            next_state, _, done, info = env.step(action)
            ep_length += 1
            ep_utility += transition_novelty(state, action, next_state, model=model)
            state = next_state

            if ep_length % model_train_freq == 0:
                model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=ep_length, mode='explore')
                mdp = None
                agent = None

        achieved_utilities.append(ep_utility)
        _log.info(f"{ep_idx}\tplanning utility: {ep_utility}")

    env.close()

    _run.result = np.mean(achieved_utilities)
    _log.info(f"average planning utility: {np.mean(achieved_utilities)}")

    return np.mean(achieved_utilities)


@ex.capture
def checkpoint(buffer, step_num, dump_dir, _run):
    buffer_file = f'{dump_dir}/{step_num}.buffer'
    with gzip.open(buffer_file, 'wb') as f:
        pickle.dump(buffer, f)
    _run.add_artifact(buffer_file)


@ex.capture
def evaluate_agent(agent, step_num, n_eval_episodes):
    env = get_env()
    env = TorchEnv(env)
    ep_returns = []
    for ep_idx in range(n_eval_episodes):
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


"""
Main Functions
"""


@ex.capture
def do_max_exploration(seed, action_noise_stdev, n_exploration_steps, n_warm_up_steps,  
                       agent_train_freq, model_train_freq, 
                       n_explore_rollout_steps, explore_rollout_freq,
                       policy_batch_size, agent_batch_size, device,
                       agent_active_updates, explore_agent_updates,
                       exploring_model_epochs, eval_freq, checkpoint_frequency, 
                       render, record, dump_dir, _config, _log, _run):

    env = get_env()
    env = TorchEnv(env)
    env.seed(seed)
    atexit.register(lambda: env.close())

    buffer = get_buffer()
    exploration_measure = get_utility_measure()

    if _config['normalize_data']:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    agent = get_policy(buffer=None, model=None, measure=None, 
                       policy_batch_size=agent_batch_size, policy_lr=3e-4, 
                       mode='explore', buffer_reuse=False)

    explore_rollout_step_num = n_explore_rollout_steps

    if record:
        video_filename = f"{dump_dir}/exploration_0.mp4"
        state = env.reset(filename=video_filename)
    else:
        state = env.reset()

    for step_num in range(1, n_exploration_steps + 1):
        # real env rollout
        if step_num > n_warm_up_steps:
            if step_num % explore_rollout_freq == 0:
                explore_rollout_step_num = 0
            if explore_rollout_step_num < n_explore_rollout_steps:
                with torch.no_grad():
                    action = explore_agent(state)
                explore_rollout_step_num += 1
                if action_noise_stdev:
                    action = action + action_noise_stdev * torch.randn(action.shape)
            else:
                with torch.no_grad():
                    action = agent(state)
        else:
            action = env.action_space.sample()
            action = torch.from_numpy(action).float().to(device)
            action = action.unsqueeze(0)

        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, next_state)
        agent.replay.add(state, action, reward, next_state)

        if render:
            env.render()

        if done:
            _log.info(f"step: {step_num}\tepisode complete")

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

        just_finished_warm_up = (step_num == n_warm_up_steps)
        # train task policy
        if step_num % agent_train_freq == 0 or just_finished_warm_up:
            # _log.info(f"step: {step_num},\ttask_policy training")
            for _ in range(agent_active_updates):
                agent.update()

        # train dynamic model and exploration policy
        if step_num % model_train_freq == 0 or just_finished_warm_up:
            _log.info(f"step: {step_num}, \tdynamic_model training")
            model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=step_num, mode='explore')
            _log.info(f"step: {step_num}, \texplore_policy training")
            explore_agent = deepcopy(agent)
            explore_agent.set_batch_size(policy_batch_size)
            explore_agent = transfer_buffer_to_agent(buffer=buffer, 
                                                     agent=explore_agent,
                                                     model=model,
                                                     measure=exploration_measure)
            for _ in range(explore_agent_updates):
                explore_agent.update()

        # evaluate taks policy
        if (step_num % eval_freq) == 0 and step_num > n_warm_up_steps:
            avg_return = evaluate_agent(agent, step_num)
            _log.info(f"step: {step_num}, evaluate:\taverage return = {np.round(avg_return, 4)}")
            writer.add_scalar(f"evaluate_return", avg_return, step_num)

        # time_to_checkpoint = ((step_num % checkpoint_frequency) == 0)
        # if time_to_checkpoint:
        #     checkpoint(buffer=buffer, step_num=step_num)

    if record:
        _run.add_artifact(video_filename)

    return avg_return
    # return max_return
    # return max(average_performances)


@ex.capture
def do_random_exploration(seed, normalize_data, n_exploration_steps, n_warm_up_steps, eval_freq, _log):
    env = get_env()
    env.seed(seed)
    atexit.register(lambda: env.close())

    buffer = get_buffer()
    if normalize_data:
        normalizer = TransitionNormalizer()
        buffer.setup_normalizer(normalizer)

    average_performances = []
    state = env.reset()
    for step_num in range(1, n_exploration_steps + 1):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, next_state)

        if done:
            _log.info(f"step: {step_num}\tepisode complete")
            next_state = env.reset()

        state = next_state

        time_to_evaluate = ((step_num % eval_freq) == 0)
        just_finished_warm_up = (step_num == n_warm_up_steps)
        if time_to_evaluate or just_finished_warm_up:
            average_performance = evaluate_tasks(buffer=buffer, step_num=step_num)
            average_performances.append(average_performance)

    checkpoint(buffer=buffer, step_num=n_exploration_steps)

    return max(average_performances)


@ex.capture
def do_exploitation(seed, normalize_data, n_exploration_steps, buffer_file, ensemble_size, benchmark_utility, _log, _run):
    if len(buffer_file):
        with gzip.open(buffer_file, 'rb') as f:
            buffer = pickle.load(f)
        buffer.ensemble_size = ensemble_size
    else:
        env = get_env()
        env.seed(seed)
        atexit.register(lambda: env.close())

        buffer = get_buffer()
        if normalize_data:
            normalizer = TransitionNormalizer()
            buffer.setup_normalizer(normalizer)

        state = env.reset()
        for step_num in range(1, n_exploration_steps + 1):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, next_state)

            if done:
                _log.info(f"step: {step_num}\tepisode complete")
                next_state = env.reset()

            state = next_state

    if benchmark_utility:
        return evaluate_utility(buffer=buffer)
    else:
        return evaluate_tasks(buffer=buffer, step_num=0)


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
    elif random_exploration:
        return do_random_exploration()
    elif exploitation:
        return do_exploitation()

