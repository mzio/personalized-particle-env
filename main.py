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
from models.actor_critic import ActorCritic as model

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-s', '--scenario', default='simple.py',
                    help='Path of the scenario Python script')
parser.add_argument('--num_agents', default=None, type=int)
parser.add_argument('--load_agents', default='')
parser.add_argument(
    '--save_agents', default='agents-0.json')
parser.add_argument('--lr', default=1e-2, type=int,
                    help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Print for debugging')
parser.add_argument('-p', '--personalization',
                    help='Personalization setup: "variance", "remap", "none" supported')
parser.add_argument('--specific_agents', default='',
                    help='Only load specific agent(s)')
parser.add_argument('-e', '--episode_len', default=1000,
                    type=int, help='Number of timesteps per episode')
parser.add_argument('-ne', '--num_episodes', default=100,
                    type=int, help='How many episodes to run')
parser.add_argument('-r', '--render', action='store_true',
                    help='Render gridworld window')
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument('--log_interval', default=1, type=int,
                    help='Logging rate')
parser.add_argument('--save_results', default='./results/results.csv')
parser.add_argument('--save_model', default='./trained_models/model.pt')
args = parser.parse_args()

load_agents = None
if args.load_agents != '':
    load_agents = './particles/configs/' + args.load_agents + '.json'

save_agents = './particles/configs/' + args.save_agents + '.json'

if args.specific_agents != '':
    specific_agents = args.specific_agents.split(' ')
else:
    specific_agents = None

scenario = scenarios.load(args.scenario).Scenario(
    kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
    load_agents=load_agents, save_agents=save_agents,
    specific_agents=specific_agents)
# create world
world = scenario.make_world()
world.episode_len = args.episode_len

env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                       scenario.observation, info_callback=None,
                       done_callback=scenario.done, shared_viewer=True)
env.discrete_action_input = True

if args.render:
    env.render()

policies = [model(env, i, env.observation_space[i].shape[0],
                  env.action_space[0].n) for i in range(env.n)]

optimizer = optim.Adam(policies[0].parameters(), lr=args.lr)


obs_n = env.reset()
running_reward = 10

info = [['Timestep', 'Episode', 'State_x_vel', 'State_y_vel',
         'State_x_pos', 'State_y_pos', 'Action', 'Reward', 'Episode_Reward']]

total_timesteps = 0

for n in range(args.num_episodes):
    t = 0
    env.reset()
    while t < args.episode_len:
        ep_reward = 0
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
            # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        if args.debug:
            print('OBSERVATIONS: {}'.format(obs_n))
        # render all agent views
        if args.render:
            env.render()
        # display rewards
        if args.debug:
            for agent in env.world.agents:
                print(agent.name + " reward: %0.3f" %
                      env._get_reward(agent))
        policy.rewards.append(reward_n[0])
        ep_reward += reward_n[0]
        t += 1
        total_timesteps += 1
        if done_n[i] is True:
            break

        info.append([total_timesteps, n, obs_n[0][0], obs_n[0][1],
                     obs_n[0][2], obs_n[0][3],
                     act_n[0], reward_n[0], ep_reward])

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    policy.finish_episode(optimizer, args.gamma)
    # if n % args.log_interval == 0:
    print('Episode {}\tLast reward: {:.3f}\tAverage reward: {:.2f}'.format(
        n, ep_reward, running_reward))


# Save model and results
torch.save(policies[0].state_dict(), args.save_model)

with open(args.save_results, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(info)

# python main.py --scenario simple.py --p 'none' --seed {} --save_results './results/results_{}.csv' --save_model './trained_models/model_{}.pt'

# if running_reward > env.spec.reward_threshold:
#     print("Solved! Running reward is now {} and "
#           "the last episode runs to {} time steps!".format(running_reward, t))
#     break
