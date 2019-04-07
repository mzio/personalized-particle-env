# Core classes describing state and actions of agents, along with environment physics
# Inspired by https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py

import numpy as numpy


# Parent state class for various objects, e.g. agents, landmarks
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_vel = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        """TODO: Think about what else should be here"""


class Action(object):
    def __init__(self):
        self.m = None  # Movement action


# Properties and state of physical world entity
class Entity(object):
    def __init__(self):
        self.name = ''
        self.size = 0.050
        self.movable = False  # Can move / be pushed
        self.collide = True   # Allow collisions with others
        self.density = 25.0   # Affects mass / movement
        self.color = None
        self.max_speed = None
        self.accel = None
        self.initial_mass = 1.0
        self.state = EntityState()

    @property
    def mass(self):
        return self.initial_mass


# Properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# Properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.collide = True
        self.m_noise = None  # Physical motor noise
        self.m_range = 1.0   # Control range
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None  # Scripted agent behavior execution
        # Personalized action function.
        self.personalize = None


# (Multi-)agent world
class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.dim_m = 0       # Position dimensionality
        self.dim_color = 3   # Color dimensionality
        self.dt = 0.1        # Simulation timestep
        self.damping = 0.25  # Physical damping
        # Contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # Return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # Return all agents controllable by external policies, disjoint of scripted agents
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # Return all agents controlled by world scripts, disjoint of policy agents
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # Update state of the world
    def step(self):
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # Gather forces applied
        p_force = [None] * len(self.entities)
        p_roce = self.apply_action_force(p)

    # Gather agent action forces
    def apply_action_force(self, p_force):
        for i, agent in enumerate(self, agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.m.shape) * \
                    agent.m_noise if agent.m_noise else 0.
                p_force[i] = agent.personalize(agent.action.m) + noise
        return p_force

    # Gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # Sad quadratic time collision calculation
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None):
                        p_force[a] = 0.0
                if (f_b is not None):
                    if (p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # Integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # Set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # Get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # Not a collider
        if (entity_a is entity_b):
            return [None, None]  # Don't collide against itself
        # Compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # Minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # Softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
