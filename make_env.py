"""
Code for creating personalized agent environment with any of 
the scenarios found in ./scenarios/, based on OpenAI multiagent env
found here: https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py
Can be called using `env = make_env('simple')  
After making the env object, behaves similarly to OpenAI gym env  

Policies have to output actions in the form of a list for all agents
"""


def make_env(scenario_name, benchmark=False):
    '''
    Creates a PersonalAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from particles.environment import PersonalAgentEnv
    import particles.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = PersonalAgentEnv(world, scenario.reset_world, scenario.reward,
                               scenario.observation, scenario.benchmark_data)
    else:
        env = PersonalAgentEnv(world, scenario.reset_world,
                               scenario.reward, scenario.observation)
    return env
