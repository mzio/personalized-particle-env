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
from models.metalearner import MetaLearner

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
parser.add_argument('--num_iters', default=10, type=int,
                    help='Number of meta-iterations')
parser.add_argument('--num_eval_iters', default=100, type=int,
                    help='Number of evaluation iterations')
parser.add_argument('--optimizer', default='Adam')
args = parser.parse_args()

# Need to specify support models and agent configuration
assert args.specific_agents != ''
assert args.eval_agents != ''
assert args.load_agents != ''

load_agents = './particles/configs/' + args.load_agents + '.json'
support_agents = args.specific_agents.split(' ')

if args.model == 'ActorCritic':
    model = ActorCritic
elif args.model == 'Reinforce':
    model = Reinforce
else:
    raise NotImplementedError

# Predefine the observation and action spaces
metalearner = MetaLearner(model, 2, 5, K=10, num_iters=50,
                          initialize_size=len(support_agents), lr=args.lr)
metalearner.episode_len = args.episode_len

# META-TRAINING #
# Pick a random subset to train?
# np.random.seed(args.seed)
# np.random.choice(support_agents, 6)  # pick 6 to train randomly

for agent in support_agents:  # Train pre-trained models first
    scenario = scenarios.load(args.scenario).Scenario(
        kind=args.personalization, num_agents=args.num_agents, seed=args.seed,
        load_agents=load_agents, specific_agents=agent)

    scenario.sample_task = True

    world = scenario.make_world()
    world.episode_len = args.episode_len
    env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                           scenario.observation, info_callback=None,
                           done_callback=scenario.done, shared_viewer=True)
    env.discrete_action_input = True
    env.seed(args.seed)

    metalearner.env = env

    # Create new policy for updating
    metalearner.current_policy = metalearner.policy(
        0, metalearner.obs_shape, metalearner.action_shape)

    optimizer = optim.Adam(metalearner.current_policy.parameters(), lr=args.lr)
    metalearner.optimizer = optimizer

    # For every training iteration, save trajectory and get updates
    for n in range(args.num_iters):
        metalearner.train(args.k, n + 1)  # +1 because save after an update

    scenario.sample_task = True  # Now change the entity type

# Model should have lists of trajectories and policies now, indexed by training iteration
total_iters = np.array(range(args.num_iters)) + 1
# only get first and last updates?
metalearner.calculate_distances(iterations=[1, args.num_iters])
# Initially have these. Should we calculate divergences between all pretrained models?
# After doing that, can see which policies are close are far, and also sort of bin them together in an association list
# Then after doing this, during evaluation, we want to calculate again given some new trajectory

# EVALUATION
eval_agents = args.eval_agents.split(' ')

meta_info = [['Timestep', 'Episode', 'Episode_Reward', 'Eval_Agent']]
info = [['Timestep', 'Episode', 'State_x_pos', 'State_y_pos',
         'Action', 'Relative Reward', 'Episode_Reward', 'Eval_Agent']]

for agent in eval_agents:
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

    if args.render:
        env.render()

    obs_n = env.reset()
    running_reward = -1.0

    total_timesteps = 0
    episode_ix = 0

    # num_episodes = int(args.num_episodes / args.k)

    for n in range(args.num_eval_iters):  # 100? Number of updates
        if n == 0:
            policy = model(0, 2, 5)  # Initiate new policy
            optimizer = optim.Adam(policy.parameters(), lr=args.lr)
            for update in range(args.k):
                metalearner.sample(policy)
            metalearner.adapt(policy, optimizer, args.k)
            trajectory, ep_reward = metalearner.sample(
                policy)  # Get new trajectory and ep_reward
            total_timesteps += args.k * args.episode_len

            meta_info.append([total_timesteps, episode_ix, ep_reward, agent])

            episode_ix += 1

            policies = metalearner.get_updated_policies(
                args.k, policy, trajectory, 1)
            rewards = [policy['reward'] for policy in policies]
            # Just pick highest reward policy for now
            policy_ix = np.array(rewards).argsort[0]
            policy = policies[policy_ix]['policy']
            optimizer = policies[policy_ix]['optimizer']

            env.reset()
        else:
            # After this fast adaptation stage, revert to VPG
            for update in range(args.k):
                t = 0
                while t < args.episode_len:
                    ep_reward = 0
                    act_n = []
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

                    try:
                        relative_reward = policy.rewards[-1] - \
                            policy.rewards[-2]
                    except:
                        relative_reward = 0

                    if total_timesteps % args.log_interval == 0:
                        info.append([total_timesteps, episode_ix, obs_n[0][0], obs_n[0][1],
                                     act_n[0], relative_reward, ep_reward, agent])
                    if done_n[i] is True:
                        continue
                policy.finish_episode(optimizer, args.gamma)
                episode_ix += 1
                if total_timesteps % args.log_interval == 0:
                    print('Episode {}\tLast reward: {:.3f}\tAverage reward: {:.2f}'.format(
                        episode_ix, ep_reward, running_reward))
                env.reset()
                # info.append([total_timesteps, n, obs_n[0][0], obs_n[0]
                #              [1], act_n[0], relative_reward, ep_reward])
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            policy.update(optimizer, args.inner_updates)
            env.reset()

    # # Save model and results
    # print('Saving model...')
    # torch.save(policies[0].state_dict(), args.save_model)

with open(args.save_results, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(info)

meta_results_fname = 'meta_{}'.format(args.save_results)

with open(meta_results_fname, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(meta_info)


# python meta_main.py --num_iters 30 --k 10 --seed 1 --load_agents 'agents-clustered-p' --num_eval_iters 100 --model 'Reinforce' --log_interval 1 --episode_len 100 --optimizer 'Adam' --specific_agents 'PersonalAgent-0 PersonalAgent-1 PersonalAgent-3 PersonalAgent-8 PersonalAgent-9 PersonalAgent-10 PersonalAgent-13 PersonalAgent-15 PersonalAgent-16 PersonalAgent-18 PersonalAgent-21 PersonalAgent-22' --eval_agents 'PersonalAgent-2 PersonalAgent-4 PersonalAgent-5 PersonalAgent-6 PersonalAgent-7 PersonalAgent-11 PersonalAgent-12 PersonalAgent-14 PersonalAgent-17 PersonalAgent-19 PersonalAgent-20 PersonalAgent-23'

# python main.py --num_agents 10 --personalization 'variance' --load_agents 'agents_many_10-1' --seed 42 --specific_agents 'PersonalAgent-0' --model 'Reinforce' --inner_updates 10 --log_interval 1 --episode_len 1000 --num_episodes 1000 --save_results './results/results-r-1.csv' --save_model './trained_models/model-r-1.pt'

# python main.py --num_episodes 100 --p 'cluster' --seed 1 --save_results './results/results_ppe_simple_reinforce_sgd_5-1.csv' --save_model './trained_models/model_ppe_simple_reinforce_sgd_5-1.pt' --load_agents 'agents-clustered' --specific_agents 'PersonalAgent-5' --model 'Reinforce' --inner_updates 1 --log_interval 1 --episode_len 100

# python meta_train.py --num_episodes 100 --seed 1 --save_results './results/results_ppe-joint_reptile-0.csv' --save_model './trained_models/model_ppe-joint_reptile-0.pt' --load_agents 'agents-clustered-p' --specific_agents 'PersonalAgent-0 PersonalAgent-1 PersonalAgent-2 PersonalAgent-3 PersonalAgent-4 PersonalAgent-5 PersonalAgent-8 PersonalAgent-9 PersonalAgent-10 PersonalAgent-11 PersonalAgent-12 PersonalAgent-15' --model 'Reinforce' --inner_updates 10 --k 1 --log_interval 1 --episode_len 100 --optimizer 'Adam'
