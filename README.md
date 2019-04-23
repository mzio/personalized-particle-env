# Personalized Particle Environment

Code for a continuous gridworld particle environment for studying personalization in RL. Inspired from and sits on top of the codebase for OpenAI's [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs).

## Usage  
Continuous particle environment where particles respond to actions ("interventions") in different ways. Continuous state space defined by the current coordinates of the agent particle. Goal is to move from starting coordinates to target position coordinates. Per timestep reward is calculated by Euclidean distance to the target position.  

## Personalization?  
An anticipatory tldr; we define personalization to be any process where the act of learning an optimal policy for a specific agent type leads to divergence from a base policy. We model this through the agent's interactions. Default behavior includes moving one unit up in the direction specified (up, down, left, right). Instead, upon initiation, we randomly initialize mappings from these controls to a different transition, e.g. "up" maps to moving in the direction (0.5, 1). This accordingly leads to different state and reward transition dynamics.

## Preliminary Structure  

### `main.py`  
Main executable. Specifies the scenario, loads personalized agents, and trains policy.  

### `interactive.py`  
Interactive version of the environment. Controllable with up, down, left, right arrow keys.  

### `enjoy.py`  
Load pre-trained policy to visualize results.  

### `particles/` 
Contains code for environment, specifying personalized agents, and other back-end parts related to making the environment run. Inspired and borrows code from OpenAI's [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs).  

#### `core.py`  
Defines classes for all the objects involved in the environment. Of specific interest is the `Population` class, which loads randomly generated personalized agents.
