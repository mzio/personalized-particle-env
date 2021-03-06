# Simple scenario - agent needs to get to the only landmark

import argparse
import numpy as np
from particles.core import World, Agent, Population, Landmark
from particles.scenario import BaseScenario
# import time


class Scenario(BaseScenario):
    def __init__(self, kind, seed, num_agents=None, load_agents=None,
                 save_agents=None, include_default=False,
                 specific_agents=None):
        # Intialize by creating population of potnetial agents
        self.population = Population(
            num_agents=num_agents, personalization=kind, seed=seed,
            load_agents=load_agents, save_agents=save_agents,
            include_default=include_default, specific_agents=specific_agents)
        self.num_agents = 1  # Number of agents at a time
        self.seed = seed  # Reproducibility
        self.random_start = False
        self.random_landmarks = False
        # Deterministic seeds over time for random reloading
        self.pos_seeds = [seed]

    def make_world(self):
        world = World()
        # Add agents
        world.agents = self._sample_agents()
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.state.p_pos = [0., 0.]
            agent.state.p_vel = np.zeros(world.dim_p)
        # Add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark-{}'.format(i)
            landmark.collide = True
            landmark.movable = False
            landmark.state.p_pos = [0.25, 0.25]
            landmark.state.p_vel = np.zeros(world.dim_p)
        # Make initial conditions
        self.reset_world(world)
        return world

    def _sample_agents(self):
        """Sample agents to be present in current world episode"""
        try:
            selected_agents = np.random.choice(
                self.population.agents, size=self.num_agents, replace=False)
        except ValueError:  # Expecting larger sample than population
            selected_agents = self.population.agents
        return selected_agents

    def reset_world(self, world):
        world.agents = self._sample_agents()
        # Because coloring
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.75, 0.75])

        # Logging rewards and updating for next round
        if self.random_landmarks or self.random_start:
            self.pos_seeds.append(self.pos_seeds[-1] + 1)
        # print('Agent reward(s): {}'.format(
        #     [self.reward(a, world) for a in world.agents]))
        # print(self.pos_seeds[-1])

        world.timesteps = 0

        # Set random initial states
        if self.random_start:
            np.random.seed(self.pos_seeds[-1])
        for agent in world.agents:
            start_pos = (np.random.uniform(-1, 1, world.dim_p)
                         if self.random_start else np.array([0., 0.]))
            agent.state.p_pos = start_pos
            agent.state.p_vel = np.zeros(world.dim_p)
            # print('Agent Pos: {}'.format(agent.state.p_pos))

        if self.random_landmarks:
            np.random.seed(self.pos_seeds[-1])
        for landmark in world.landmarks:
            start_pos = (np.random.uniform(-1, 1, world.dim_p)
                         if self.random_landmarks else np.array([+0.75, +0.75]))
            landmark.state.p_pos = start_pos
            landmark.state.p_vel = np.zeros(world.dim_p)
            # print('Landmark Pos: {}'.format(landmark.state.p_pos))

    def _get_distance(self, agent, world):
        # Calculate euclidean distance
        return np.sum(np.square(np.array(agent.state.p_pos) -
                                np.array(world.landmarks[0].state.p_pos)))

    def reward(self, agent, world):
        # Euclidean distance reward
        dist2 = self._get_distance(agent, world)
        # print((world.episode_len - world.timesteps) / world.episode_len)
        if dist2 < 0.001:  # Faster agents are rewarded more
            return (-dist2 + (world.episode_len - world.timesteps) / world.episode_len)
        return -dist2

    def observation(self, agent, world):
        # Get positions of all entities in this agent's reference frame
        return np.concatenate([agent.state.p_pos])
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(np.array(entity.state.p_pos) -
                              np.array(agent.state.p_pos))
        return np.concatenate([agent.state.p_vel] + entity_pos + [agent.state.p_pos])

    def done(self, agent, world):
        # If agent reaches reward, return done
        if self._get_distance(agent, world) < 0.001:
            return True
        return False


# If call from terminal, generate population of agents beforehand
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_agents', default=None, type=int)
    parser.add_argument('-p', '--personalization',
                        help='Personalization setup: "variance", "remap", "none" supported')
    parser.add_argument(
        '--save_agents', default='agents-0.json')
    parser.add_argument('--seed', default=42, type=int,
                        help='Randomization seed')
    save_agents = '../configs/' + args.save_agents

    scenario = Scenario(kind=args.kind, seed=args.seed,
                        num_agents=args.num_agents, save_agents=save_agents)
    print('{} agent(s) generated!'.format(args.num_agents))
    for config in scenario.population.saved_agent_configs:
        print(config)
