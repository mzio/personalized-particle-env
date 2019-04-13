#!/usr/bin/env python
import argparse
import os
import sys
import inspect

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import particles.scenarios as scenarios
from particles.policy import InteractivePolicy
from particles.environment import PersonalAgentEnv


# Script for interacting with environments

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py',
                        help='Path of the scenario Python script.')
    parser.add_argument('-d', '--debug', help='Print for debugging.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                           scenario.observation, info_callback=None, shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    for n in range(100):
        x = 0
        while x < 200:
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            print(obs_n)
            # render all agent views
            env.render()
            # display rewards
            if args.debug:
                for agent in env.world.agents:
                    print(agent.name + " reward: %0.3f" %
                          env._get_reward(agent))
            x += 1
        env.reset()
