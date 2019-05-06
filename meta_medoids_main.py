# Main file to run for Meta-RL

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
from models.reinforce import Reinforce
from models.metamedoids import MetaMedoids

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
parser.add_argument('--num_medoids', default=6, type=int,
                    help='Number of medoids to cluster to')
parser.add_argument('--medoid_size', default=10, type=int,
                    help='Number of points to cluster with')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Print for debugging')
parser.add_argument('--specific_agents', default='',  # Want to load the designated training agents here
                    help='Only load specific agent(s)')
parser.add_argument('--eval_agents', default='',  # Want to load the designated evaluation agents here
                    help='Load agent type(s) for evaluation')
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
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size during regular updates')
parser.add_argument('--num_iters', default=100, type=int,
                    help='Number of meta-iterations')
parser.add_argument('--num_eval_iters', default=100, type=int,
                    help='Number of evaluation iterations')
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('-ro', '--replace_optimizer', action='store_true',
                    help='If true, replace optimizer when loading model')
args = parser.parse_args()

# Need to specify support models and agent configuration
assert args.specific_agents != ''
assert args.eval_agents != ''
assert args.load_agents != ''

load_agents = './particles/configs/' + args.load_agents + '.json'
support_agents = args.specific_agents.split(' ')
eval_agents = args.eval_agents.split(' ')

if args.model == 'ActorCritic':
    model = ActorCritic
elif args.model == 'Reinforce':
    model = Reinforce
else:
    raise NotImplementedError

# Predefine the observation and action spaces
metalearner = MetaMedoids(model, 2, 5, K=10, num_iters=50,
                          num_medoids=args.num_medoids,
                          medoid_size=args.medoid_size,
                          lr=args.lr)
metalearner.episode_len = args.episode_len

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

metalearner.env = env

# META-TRAINING #
update_counter = 1

if args.debug:
    print('Meta-training...')
    print('Load first batch')

for n in range(args.medoid_size):  # First group, just initialize with new policies
    scenario.sample_task = True
    metalearner.current_policy = metalearner.policy(
        0, metalearner.obs_shape, metalearner.action_shape)
    optimizer = optim.Adam(
        metalearner.current_policy.parameters(), lr=args.lr)
    metalearner.optimizer = optimizer
    # Add the first batch to memory, can compute distance matrix on this
    metalearner.train(args.k, update_counter, save=True)
    if args.debug:
        print('Updated learner with batch 1')
    # Maybe redundant, but make sure new entity type is introduced
    scenario.sample_task = True


# Update distances
metalearner.calculate_distances(iterations=[update_counter])

if args.debug:
    print('Initial distances calculated!')

metalearner.update_medoids(update_counter, k=args.num_medoids)

if args.debug:
    print('Initial medoids calculated!')

update_counter += 1

remaining_iterations = args.num_iters - args.medoid_size

num_obs = int(args.k / 2)  # Try this for now

for n in range(remaining_iterations):
    scenario.sample_task = True
    rewards = []
    if args.debug:
        print('Number of medoids: {}'.format(len(metalearner.medoid_policies)))
    for mix, policy in enumerate(metalearner.medoid_policies):
        if args.debug:
            print('Number of medoids: {}'.format(
                len(metalearner.medoid_policies)))
        metalearner.current_policy = policy['policy']
        metalearner.optimizer = policy['optimizer']
        if args.debug:
            print('Metalearner policy {} loaded'.format(mix))
        _, r = metalearner.train(num_obs, update_counter, save=False)
        if args.debug:
            print('Number of medoids after train: {}'.format(
                len(metalearner.medoid_policies)))
        if args.debug:
            print('Initial trajectory completed for task {} and policy {}'.format(n, mix))
        rewards.append(r)
    max_reward_ix = np.array(rewards).argsort()[-1]
    selected_policy = metalearner.medoid_policies[max_reward_ix]
    if args.debug:
        print('Best medoid policy identified!')
    metalearner.current_policy = selected_policy['policy']
    metalearner.optimizer = selected_policy['optimizer']
    if args.debug:
        print('Running medoid policy on new entity...')
    metalearner.train(args.k - num_obs, update_counter,
                      save=True)  # Do remaining updates

    if args.debug:
        print('Metalearner updated on batch {}'.format(update_counter))
    # Calculate distances if batch size is different
    if (n % args.medoid_size == 0) and (n != 0):
        if args.debug:
            print('Updating medoids...')
        # Update distances
        metalearner.calculate_distances(iterations=[update_counter])
        if args.debug:
            print('Distances calculated!')
        metalearner.update_medoids(update_counter, k=args.num_medoids)
        if args.debug:
            print('Medoids updated!')
        update_counter += 1

# Now metalearner.medoid_policies should have the medoid policies #

# EVALUATION #
if args.debug:
    print('Evaluating meta-learner...')
eval_agents = args.eval_agents.split(' ')

# meta_info = [['Timestep', 'Episode', 'Episode_Reward', 'Eval_Agent']]
info = [['Timestep', 'Episode', 'State_x_pos', 'State_y_pos',
         'Action', 'Relative Reward', 'Episode_Reward', 'Eval_Agent']]

for agent in eval_agents:
    if args.debug:
        print('Evaluating on agent: {}'.format(agent))
    scenario = scenarios.load('simple.py').Scenario(
        kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
        load_agents=load_agents, specific_agents=agent)
    # create world
    world = scenario.make_world()
    world.episode_len = args.episode_len

    env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                           scenario.observation, info_callback=None,
                           done_callback=scenario.done, shared_viewer=True)
    env.discrete_action_input = True
    env.seed(args.seed)
    metalearner.env = env

    metalearner.env = env

    if args.render:
        env.render()

    obs_n = env.reset()
    running_reward = -1.0

    total_timesteps = 0
    episode_ix = 0

    # num_episodes = int(args.num_episodes / args.k)

    policy_rewards = [[] for _ in range(metalearner.num_medoids)]

    # Number of updates we're allowed to explore with
    # Given n clusters, have k shots. Follow up w/ more complex algo like bandits?
    # Right now just take highest average reward from pulls sort of equally given
    for k in range(args.k):
        ix = k % metalearner.num_medoids
        policy = metalearner.medoid_policies[ix]
        metalearner.current_policy = policy['policy']
        metalearner.optimizer = policy['optimizer']
        _, ep_reward = metalearner.sample(metalearner.current_policy)
        policy_rewards[ix].append(ep_reward)

    # Get max reward medoid policy
    max_reward_ix = np.array(policy_rewards).mean(axis=0).argsort()[-1]
    selected_policy = metalearner.medoid_policies[max_reward_ix]
    policy = selected_policy['policy']
    if args.replace_optimizer:
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    else:
        optimizer = selected_policy['optimizer']

    for n in range(args.num_eval_iters - args.k):  # 100? Number of updates, continue
        # After this fast adaptation stage, revert to VPG
        for update in range(args.batch_size):
            t = 0
            while t < args.episode_len:
                ep_reward = 0
                act_n = []
                act_n.append(policy.action(obs_n[0]))
                # step environment
                obs_n, reward_n, done_n, _ = env.step(act_n)
                # if args.debug:
                #    print('OBSERVATIONS: {}'.format(obs_n))
                # render all agent views
                if args.render:
                    env.render()
                # display rewards
                # if args.debug:
                  #   for agent in env.world.agents:
                    #     print(agent.name + " reward: %0.3f" %
                    #        env._get_reward(agent))
                policy.rewards.append(reward_n[0])
                ep_reward += reward_n[0]
                t += 1
                total_timesteps += 1

                try:
                    relative_reward = policy.rewards[-1] - \
                        policy.rewards[-2]
                except:
                    relative_reward = 0

                if total_timesteps % args.log_interval == 0:
                    info.append([total_timesteps, episode_ix, obs_n[0][0], obs_n[0][1],
                                 act_n[0], relative_reward, ep_reward, agent])
                if done_n[0] is True:
                    continue
            policy.finish_episode(optimizer, args.gamma)
            episode_ix += 1
            if total_timesteps % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.3f}'.format(
                    episode_ix, ep_reward))
            env.reset()
        policy.update(optimizer, args.batch_size)
        env.reset()

    # # Save model and results
    # print('Saving model...')
    # torch.save(policies[0].state_dict(), args.save_model)

with open(args.save_results, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(info)

# meta_results_fname = '{}-meta.csv'.format(args.save_results.split('.csv')[0])

# with open(meta_results_fname, 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(meta_info)


# python meta_medoids_main.py --medoid_size 10 --num_medoids 5 --k 10 --seed 1 --load_agents 'agents-clustered-p' --num_eval_iters 10 --model 'Reinforce' --log_interval 1 --episode_len 100 --optimizer 'Adam' --specific_agents 'PersonalAgent-0 PersonalAgent-1 PersonalAgent-3 PersonalAgent-8 PersonalAgent-9 PersonalAgent-10 PersonalAgent-13 PersonalAgent-15 PersonalAgent-16 PersonalAgent-18 PersonalAgent-21 PersonalAgent-22' --eval_agents 'PersonalAgent-2 PersonalAgent-4 PersonalAgent-5 PersonalAgent-6 PersonalAgent-7 PersonalAgent-11 PersonalAgent-12 PersonalAgent-14 PersonalAgent-17 PersonalAgent-19 PersonalAgent-20 PersonalAgent-23'
