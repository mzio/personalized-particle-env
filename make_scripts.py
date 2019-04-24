for p in range(16):
    name = 'PersonalAgent-' + str(p) if p < 15 else ''
    for i in range(5):
        fname = './run_scripts/ppe_simple_reinforce_{}-{}.sh'.format(p, i)
        job_id = 'ppe_simple_reinforce_{}-{}'.format(p, i)
        with open(fname, 'w') as rsh:
            rsh.write('''\
#!/bin/bash
#SBATCH -J {}  # Job name
#SBATCH -p fas_gpu               # Partition to submit to
#SBATCH --gres=gpu:1             # Number of GPUs to use
#SBATCH -t 0-07:00               # Runtime
#SBATCH --mem=4000               # Memory
#SBATCH -o output_{}_%j.o            # File that STDOUT writes to
#SBATCH -e error_{}_%j.e            # File that STDERR writes to

## Setup environment ##
module load Anaconda3/5.0.1-fasrc01 cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
source activate prl_env

python main.py \
--scenario simple.py \
--num_episodes 1000 \
--p 'variance' --seed {} \
--save_results './results/results_{}.csv' \
--save_model './trained_models/model_{}.pt' \
--load_agents 'agents-clustered' \
--specific_agents '{}' \
--model 'Reinforce' \
--inner_updates 10 \
--log_interval 1 \
--episode_len 1000
'''.format(job_id, job_id, job_id, i, job_id, job_id, name))
