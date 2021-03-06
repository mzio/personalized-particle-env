# Core classes describing state and actions of agents, along with environment physics
# Inspired by https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py

import numpy as np
import json


# Parent state class for various objects, e.g. agents, landmarks
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_vel = None


# So far just a wrapper for the above
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        """TODO: Think about what else should be here"""


class Action(object):
    def __init__(self):
        self.m = None  # Movement action


# Properties and state of physical world entity
class Entity(object):
    def __init__(self, name=None):
        self.name = name
        self.size = 0.015
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
    def __init__(self, name=None, mapping=None):
        super(Agent, self).__init__()
        self.movable = True
        self.collide = True
        self.m_noise = None  # Physical motor noise
        self.m_range = 1.0   # Control range
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None  # Scripted agent behavior execution
        self.name = name
        # Personalized action function.
        self.personalize = lambda x: mapping[x]
        # Used for identification
        self.mapping = self._polarize(mapping)
        self.cluster = None

    def _polarize(self, mapping):
        # Convert [x, y] to polar representation
        polar_mapping = {}
        for k, v in mapping.items():
            polar_mapping[k] = np.arctan2(v[1], v[0])
        return polar_mapping


# Population of agents
class Population(object):
    def __init__(self, num_agents=None, personalization='variance', seed=None,
                 load_agents=None, save_agents='agents.json', include_default=False,
                 specific_agents=None, num_clusters=None):
        """
        Defines a population of agents, which may have personalized reactions to the input actions
        :load_agents: if pre-specified, just load the agents from a .json file
        :save_agents: specifies where to save configs to
        :specified_agent: list of specific agents (by name) to load
        :personalization: 'variance', 'remap', 'cluster', 'none' 
        :num_clusters: num of clusters for 'cluster' personalization
        """
        super(Population, self).__init__()
        self.num_agents = len(  # Used to set seeds
            specific_agents) if specific_agents else num_agents
        self.agents = []
        self.saved_agent_configs = []

        if load_agents:
            # load_agents is a json that holds agent ids, their color, and mapping
            with open(load_agents, 'r') as f:
                loaded_agents = json.load(f, object_hook=self._convert_keys)

            check = specific_agents if specific_agents else [
                a['name'] for a in loaded_agents]

            for a in loaded_agents:
                if a['name'] in check:
                    agent = Agent(name=a['name'], mapping=a['mapping'])
                    agent.color = a['color']
                    agent.cluster = a['cluster']
                    agent.mapping = a['mapping_angles']
                    self.agents.append(agent)
        else:
            assert self.num_agents is not None
            # Hard code possible remaps
            self.remaps = [[-1., -1.], [-1., 0.], [-1., 1.], [0., -1.],
                           [0., 1.], [1., -1.], [1., 0.], [1., 1.]]
            self.cluster_remaps = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
            self.colors = [[1., 0, 0], [1, 1, 0], [
                0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]]
            c = 0  # Used for clustering
            num_groups = self.num_agents / num_clusters
            for i in range(self.num_agents):
                if personalization == 'cluster':
                    assert num_clusters is not None
                    if i % num_groups == 0:
                        c += 1
                    # c = i % num_clusters  # assign clusters (should change this so easier to keep track of)
                    mapping = self.get_personalization(
                        seed=i, kind='cluster', cluster=c)
                else:
                    if i == 0 and include_default:
                        mapping = self.get_personalization(seed=i, kind='none')
                    else:
                        mapping = self.get_personalization(
                            seed=i, kind=personalization)
                name = 'PersonalAgent-{}'.format(i)
                agent = Agent(name=name, mapping=mapping)
                try:  # Cluster
                    agent.cluster = c
                except:
                    agent.cluster = None
                # Why not
                np.random.seed(i)
                agent.color = self.colors[i % len(self.colors)]
                self.agents.append(agent)
                self.saved_agent_configs.append(
                    {'name': name, 'mapping': mapping, 'color': agent.color,
                     'cluster': agent.cluster, 'mapping_angles': agent.mapping})

            if save_agents:
                with open(save_agents, 'w') as f:
                    # print(json.dumps(saved_agent_configs))
                    json.dump(self.saved_agent_configs, f)

    def get_personalization(self, seed, kind='variance', cluster=None):
        """
        Maps default 1-hot 4-dim input to a variation   
        By default mappings are independent of each other.
        :kind: 'variance' or 'remap' or 'none'
        - 'variance': Maps 0 value to uniform between -1 and 1, i.e. [1, 0] -> [1, Unif(-1, 1)]
        - 'remap': Remaps controls to any untaken [x, y] pairing where x, y in {-1, 0, 1}
        - 'none': Default controls
        :cluster: latent variable that determines cluster. Used to help with inherent clusters.
        """
        if kind == 'variance':
            np.random.seed(seed)
            mapping = {1: [-1., np.random.uniform(-1, 1)],  # [-1, 0]
                       2: [+1., np.random.uniform(-1, 1)],  # [+1, 0]
                       3: [np.random.uniform(-1, 1), -1.],  # [0, -1]
                       4: [np.random.uniform(-1, 1), +1.]}  # [0, +1]

        elif kind == 'remap':
            np.random.seed(seed)
            ix_n = np.random.choice(
                len(self.remaps), size=4, replace=False)
            mapping = {}
            for k, v in enumerate(self.remaps[ix_n]):
                mapping[k + 1] = v

        elif kind == 'cluster':
            assert cluster is not None
            mapping = {}
            np.random.seed(cluster * 10)  # First initialize clusters
            ix_n = np.random.choice(
                len(self.cluster_remaps), size=4, replace=False)
            mapping = {}
            for k, v in enumerate(np.array(self.cluster_remaps)[ix_n]):
                mapping[k + 1] = v
            mapping = self._randomize(mapping, seed=seed)

        elif kind == 'none':
            mapping = {1: [-1., 0.], 2: [1., 0.], 3: [0., -1.], 4: [0., 1.]}

        else:
            print('Invalid kind of personalization specified.')
            raise NotImplementedError
        mapping[0] = [0., 0.]
        return mapping

    def _convert_keys(self, x):
        """Convert string keys to integers when loading json agents file"""
        if isinstance(x, dict):
            try:
                return {int(k): v for k, v in x.items()}
            except:
                return x
        return x

    def _randomize(self, mapping, seed, b=0.5):
        """
        Helper function for further variance
        - Currently uses uniform distribution (b is bound)
        """
        np.random.seed(seed)
        map_ = {}
        for k, v in mapping.items():
            x, y = v
            if x == 0:
                map_[k] = [np.random.uniform(-b, b), y]
            elif y == 0:
                map_[k] = [x, np.random.uniform(-b, b)]
        return map_


# (Multi-)agent world
class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        # self.dim_m = 0       # Position dimensionality
        # position dimensionality
        self.dim_p = 2
        self.dim_color = 3   # Color dimensionality
        self.dt = 0.1        # Simulation timestep
        self.damping = 0.25  # Physical damping
        # Contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.timesteps = 0
        self.episode_len = None

    # Return all entities in the world
    @property
    def entities(self):
        return np.concatenate((self.agents, self.landmarks))

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
        p_force = self.apply_action_force(p_force)
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        self.timesteps += 1
        # # update agent state
        # for agent in self.agents:
        #     self.update_agent_state(agent)

    # Gather agent action forces
    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.m.shape) * \
                    agent.m_noise if agent.m_noise else 0.
                p_force[i] = np.array(agent.action.m) + noise
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

    # def update_agent_state(self, agent):
    #     # Set communication state (directly for now)
    #     if agent.silent:
    #         agent.state.c = np.zeros(self.dim_c)
    #     else:
    #         noise = np.random.randn(*agent.action.c.shape) * \
    #             agent.c_noise if agent.c_noise else 0.0
    #         agent.state.c = agent.action.c + noise

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
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
