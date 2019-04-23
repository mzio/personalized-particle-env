# Script to pre-generate agents
import argparse

from core import Population

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--num_agents', default=None, type=int)
parser.add_argument('-p', '--personalization',
                    help='Personalization setup: "variance", "remap", "cluster", "none" supported')
parser.add_argument('--num_clusters', default=None, type=int,
                    help='Number of clusters for "cluster" personalization')
parser.add_argument('--default', action='store_true',
                    help='Include default mapping')
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument(
    '--save_agents', default='agents-0.json')
args = parser.parse_args()


save_agents = './particles/configs/' + args.save_agents + '.json'

population = Population(num_agents=args.num_agents,
                        personalization=args.personalization,
                        seed=args.seed, save_agents=save_agents,
                        include_default=args.default,
                        num_clusters=args.num_clusters)

print('{} agent(s) generated!'.format(args.num_agents))
for config in population.saved_agent_configs:
    print(config)

# python generate_agents --num_agents 15 --num_clusters 5 -p 'cluster  --save_agents 'agents-clustered.json'
