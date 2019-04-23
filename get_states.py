# Run this to visualize trained policies

import gym
import csv
import numpy as np
import argparse

import torch
import torch.optim as optim

import particles
import particles.scenarios as scenarios
from particles.environment import PersonalAgentEnv

from os import listdir
from os.path import isfile, join

from models.reinforce import Reinforce

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-s', '--scenario', default='simple.py',
                    help='Path of the scenario Python script')
parser.add_argument('--load_agents', default='')
parser.add_argument('--num_agents', default=None, type=int)
parser.add_argument('-d', '--debug', action='store_true',
                    help='Print for debugging')
parser.add_argument('-p', '--personalization',
                    help='Personalization setup: "variance", "remap", "none" supported')
parser.add_argument('-e', '--episode_len', default=100,
                    type=int, help='Number of timesteps per episode')
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument('--log_interval', default=1, type=int,
                    help='Logging rate')
parser.add_argument('--specific_agents', default='',
                    help='Only load specific agent(s)')
parser.add_argument('--trained_models', default='./trained_models/',
                    help='Trained model name for visualization')
parser.add_argument('--num_states', default=5,
                    help='Number of states to collect')
parser.add_argument('--save_results', default='states.csv',
                    help='Store states')
args = parser.parse_args()

fpath = args.trained_models

trained_models = [join(fpath, f) for f in listdir(
    fpath) if (isfile(join(fpath, f))) and ('.pt' in f)]


load_agents = None
if args.load_agents != '':
    load_agents = './particles/configs/' + args.load_agents + '.json'

states = [['State_x_pos', 'State_y_pos', 'Model']]

mod = int(args.episode_len / args.num_states)

for model in trained_models:
    print('Collect states for {}...'.format(model))
    num = int(model.split('-')[0][-1])
    specific_agents = 'PersonalAgent-{}'.format(num)

    scenario = scenarios.load(args.scenario).Scenario(
        kind=args.personalization, num_agents=1, seed=num,
        load_agents=load_agents, save_agents=None,
        specific_agents=specific_agents)
    # create world
    world = scenario.make_world()
    world.episode_len = args.episode_len

    env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                           scenario.observation, info_callback=None,
                           done_callback=scenario.done, shared_viewer=True)
    env.discrete_action_input = True

    policies = [Reinforce(env, i, env.observation_space[i].shape[0],
                          env.action_space[0].n) for i in range(env.n)]

    policies[0].load_state_dict(torch.load(model))

    eps = np.finfo(np.float32).eps.item()

    obs_n = env.reset()
    running_reward = -1

    total_timesteps = 0

    for n in range(args.episode_len):
        t = 0
        act_n = []
        ep_reward = 0
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        obs_n, reward_n, done_n, _ = env.step(act_n)

        if t % mod == 0:
            state = list(obs_n[0])
            state.append(num)
            states.append(state)

        t += 1

        if done_n[i] is True:
            break

with open(args.save_results, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(states)

# python get_states.py --scenario simple.py --p 'none' --seed 0 --trained_models './trained_models/' --load_agents 'agents_many_10-1'

