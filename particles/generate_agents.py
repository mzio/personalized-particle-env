# Script to pre-generate agents
import argparse

from core import Population

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--num_agents', default=None, type=int)
parser.add_argument('-p', '--personalization',
                    help='Personalization setup: "variance", "remap", "none" supported')
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument(
    '--save_agents', default='agents-0.json')
args = parser.parse_args()


save_agents = './particles/configs/' + args.save_agents + '.json'

population = Population(num_agents=args.num_agents,
                        personalization=args.personalization,
                        seed=args.seed, save_agents=save_agents)

print('{} agent(s) generated!'.format(args.num_agents))
for config in population.saved_agent_configs:
    print(config)
