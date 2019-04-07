# Simple scenario - agent needs to get to the only landmark

import numpy as np
from particles.core import World, Agent, Population, Landmark
from particlles.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        # Intialize by creating population of potnetial agents
        self.population = Population(10, personalization='variance')
        self.num_agents = 1  # Number of agents at a time
        self.seed = 42  # Reproducibility

    def make_world(self):
        world = World()
        # Add agents
        world.agents = self._sample_agents()
        for i, agent in enumerate(world.agents):
            agent.collide = False
            agent.state.p_pos = [-0.5, -0.5]
            agent.state.p_vel = np.zeros(world.dim_p)
        # Add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark = enumerate(world.landmarks):
            landmark.name = 'landmark-{}'.format(i)
            landmark.collide = False
            landmark.movable = False
            landmark.state.p_pos = [0.5, 0.5]
            landmark.state.p_vel = np.zeros(world.dim_p)
        # Make initial conditions
        self.reset_world(world)
        return world

    def _sample_agents(self, seed=None):
        """Sample agents to be present in current world episode"""
        if seed:
            np.random.seed(seed)
        try:
            selected_agents = np.random.choice(
                self.population.agents, size=self.num_agents, replace=False)
        except ValueError:  # Expecting larger sample than population
            selected_agents = self.population.agents
        return selected_agents

    def reset_world(self, world, random_start=False, random_landmarks=True):
        world.agents = self._sample_agents()
        # Because coloring
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

        # Set random initial states
        for agent in world.agents:
            start_pos = (np.random.uniform(-1, 1, world.dim_p)
                         if random_start else [-0.5, -0.5])
            agent.state.p_pos = start_pos
            agent.state.p_vel = np.zeros(world.dim_p)

        for landmark in world.landmarks:
            start_pos = (np.random.uniform(-1, 1, world.dim_p)
                         if random_start else [+0.5, +0.5])
            landmark.state.p_pos = start_pos
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Euclidean distance reward
        dist2 = np.sum(np.square(agent.state.p_pos -
                                 world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
