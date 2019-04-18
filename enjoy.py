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
parser.add_argument('--trained_model', default='model_0.pt',
                    help='Trained model name for visualizaiton')
args = parser.parse_args()

load_agents = None
if args.load_agents != '':
    load_agents = './particles/configs/' + args.load_agents + '.json'

if args.specific_agents != '':
    specific_agents = args.specific_agents.split(' ')
else:
    specific_agents = None

scenario = scenarios.load(args.scenario).Scenario(
    kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
    load_agents=load_agents, save_agents=None,
    specific_agents=specific_agents)
# create world
world = scenario.make_world()
world.episode_len = args.episode_len

env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                       scenario.observation, info_callback=None,
                       done_callback=scenario.done, shared_viewer=True)
env.discrete_action_input = True

env.render()

policies = [Reinforce(env, i, env.observation_space[i].shape[0],
                      env.action_space[0].n) for i in range(env.n)]

policies[0].load_state_dict(torch.load(
    './trained_models/' + args.trained_model))

eps = np.finfo(np.float32).eps.item()

obs_n = env.reset()
running_reward = 10

rewards = [['Episode', 'Reward']]

total_timesteps = 0

while True:
    t = 0
    env.reset()
    while t < args.episode_len:
        act_n = []
        ep_reward = 0
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        obs_n, reward_n, done_n, _ = env.step(act_n)

        env.render()

        if done_n[i] is True:
            env.reset()
        t += 1
    print('Reward: {}'.format(reward_n[0]))
    env.reset()

# python enjoy.py --scenario simple.py --p 'none' --seed 0 --trained_model 'model_0.pt'
