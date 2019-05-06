# Main file to run for meta learning?

"""
Argument guide  
* K > 1, inner_updates = 1, optim = SGD -> Reptile
* K = 1, inner_updates = 10, optim = Adam -> Adam Joint  
* K = 1, inner_updates = 1, optim = SGD -> Reptile Joint  
* K = 1, inner_updates = 1, optim = Adam -> Adam subjoint?

"""

import gym
import csv
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable

import particles
import particles.scenarios as scenarios
from particles.environment import PersonalAgentEnv

from models.actor_critic import ActorCritic
from models.reinforce import Reinforce  # Should be consistent here?

parser = argparse.ArgumentParser(description=None)

parser.add_argument('-s', '--scenario', default='meta_simple.py',
                    help='Path of the scenario Python script')
parser.add_argument(
    '--model', default='Reinforce')
parser.add_argument('--num_agents', default=1, type=int)
parser.add_argument('-p', '--personalization', default='cluster',
                    help='Personalization setup: "variance", "remap", "cluster", "none" supported')
parser.add_argument('--load_agents', default='')
parser.add_argument(
    '--save_agents', default='agents-0.json')
parser.add_argument('--lr', default=1e-2, type=int,
                    help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--inner_updates', type=int, default=10,
                    help='Number of rollouts per batch')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Print for debugging')
parser.add_argument('--specific_agents', default='',  # Want to load the designated training agents here
                    help='Only load specific agent(s)')
parser.add_argument('-e', '--episode_len', default=100,
                    type=int, help='Number of timesteps per episode')
parser.add_argument('-ne', '--num_episodes', default=500,
                    type=int, help='How many episodes to run')
parser.add_argument('-r', '--render', action='store_true',
                    help='Render gridworld window')
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument('--log_interval', default=1, type=int,
                    help='Logging rate')
parser.add_argument('--save_results', default='./results/results.csv')
parser.add_argument('--save_model', default='./trained_models/model.pt')
parser.add_argument('--k', default=10, type=int,
                    help='Number of shots allowed')
parser.add_argument('--num_updates', default=10, type=int,
                    help='Number of meta-iterations')
parser.add_argument('--optimizer', default='Adam')
args = parser.parse_args()


if args.model == 'ActorCritic':
    model = ActorCritic
elif args.model == 'Reinforce':
    model = Reinforce
else:
    raise NotImplementedError


assert args.load_agents != ''  # Need to load agents
assert args.specific_agents != ''    # Need to specify support models

load_agents = './particles/configs/' + args.load_agents + '.json'
support_agents = args.specific_agents.split(' ')


# for agent in support_agents:
#     scenario = scenarios.load(args.scenario).Scenario(
#         kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
#         load_agents=load_agents)
scenario = scenarios.load(args.scenario).Scenario(
    kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
    load_agents=load_agents, specific_agents=support_agents)

world = scenario.make_world()
world.episode_len = args.episode_len

env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                       scenario.observation, info_callback=None,
                       done_callback=scenario.done, shared_viewer=True)
env.discrete_action_input = True
env.seed(args.seed)

policies = [model(i, env.observation_space[i].shape[0],
                  env.action_space[0].n) for i in range(env.n)]

if args.optimizer == 'Adam':
    optimizer = optim.Adam(policies[0].parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(policies[0].parameters(), lr=args.lr)
else:
    raise NotImplementedError

scenario.sample_task = True  # Start off true

running_reward = -1.0

total_timesteps = 0
episode_ix = 0


def inner_train(policy, env=env):
    new_policy = model(0, env.observation_space[0].shape[0],
                       env.action_space[0].n)
    new_policy.load_state_dict(policy.state_dict())
    # new_policy.rewards = policy.rewards
    # new_policy.finish_episode(optimizer, args.gamma)
    for i in range(args.inner_updates):  # ex. 10, then do 10 times, update the average
        run_episode(new_policy)  # if 1 inner update, == SGD,
        env.reset()
    new_policy.update(optimizer, args.inner_updates)
    return new_policy


def run_episode(policy, env=env, obs=None, train=True):  # Call this K times
    t = 0
    obs_n = env.reset()
    while t < args.episode_len:
        ep_reward = 0
        act_n = []
        for i, _ in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        obs_n, reward_n, done_n, _ = env.step(act_n)
        policy.rewards.append(reward_n[0])
        ep_reward += reward_n[0]
        if obs:
            obs.append(obs_n[0])
        t += 1
    # Want to update the policy after
    if train:
        policy.finish_episode(optimizer, args.gamma)
    else:
        # Return reward
        return ep_reward


def inner_eval(policy, obs):
    meta_reward = run_episode(policy, obs=obs, train=False)
    return meta_reward


# policy = policies[0]

for n in range(args.num_updates):
    new_policy = policies[0]
    for k in range(args.k):  # if K > 1, inner_updates = 1, optim = SGD, this is Reptile
        new_policy = inner_train(new_policy)
    type_observations = []
    # Evaluate
    train_meta_reward = inner_eval(new_policy, type_observations)
    type_observations  # also updated, can compute KDE on this
    scenario.sample_task = True

    for p, new_p in zip(policies[0].parameters(), new_policy.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size()))
        p.grad.data.add_(p.data - new_p.data)
    optimizer.step()
    optimizer.zero_grad()  # why would this work?

print('Saving model...')
torch.save(policies[0].state_dict(), args.save_model)

# python main.py --num_agents 10 --personalization 'variance' --load_agents 'agents_many_10-1' --seed 42 --specific_agents 'PersonalAgent-0' --model 'Reinforce' --inner_updates 10 --log_interval 1 --episode_len 1000 --num_episodes 1000 --save_results './results/results-r-1.csv' --save_model './trained_models/model-r-1.pt'

# python main.py --num_episodes 100 --p 'cluster' --seed 1 --save_results './results/results_ppe_simple_reinforce_sgd_5-1.csv' --save_model './trained_models/model_ppe_simple_reinforce_sgd_5-1.pt' --load_agents 'agents-clustered' --specific_agents 'PersonalAgent-5' --model 'Reinforce' --inner_updates 1 --log_interval 1 --episode_len 100

# python meta_train.py --num_updates 100 --seed 1 --save_results './results/results_ppe-joint_reptile-0.csv' --save_model './trained_models/model_ppe-joint_reptile-0.pt' --load_agents 'agents-clustered-p' --specific_agents 'PersonalAgent-0 PersonalAgent-1 PersonalAgent-2 PersonalAgent-3 PersonalAgent-4 PersonalAgent-5 PersonalAgent-8 PersonalAgent-9 PersonalAgent-10 PersonalAgent-11 PersonalAgent-12 PersonalAgent-15' --model 'Reinforce' --inner_updates 10 --k 1 --log_interval 1 --episode_len 100 --optimizer 'Adam'
