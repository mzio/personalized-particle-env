# Define scenarios for which worlds are built on
# Code from https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/scenario.py

import numpy as np


class BaseScenario(object):
    # Create elements of the world
    def make_world(self):
        raise NotImplementedError

    # Create initial conditions of the world
    def reset_world(self, world):
        rasie NotImplementedError
