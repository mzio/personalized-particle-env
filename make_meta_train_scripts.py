# Support agents
agents = 'PersonalAgent-0 PersonalAgent-1 PersonalAgent-2 PersonalAgent-3 PersonalAgent-4 PersonalAgent-5 PersonalAgent-8 PersonalAgent-9 PersonalAgent-10 PersonalAgent-11 PersonalAgent-12 PersonalAgent-15'

optim = 'SGD'
k = 1
inner_iters = 1
meta_iters = 100

for i in range(5):
    fname = './run_scripts/ppe-meta_iter_{}-inner_iter_{}-k_{}-{}-{}.sh'.format(
        meta_iters, inner_iters, k, i)
    job_id = 'ppe-meta_iter_{}-inner_iter_{}-k_{}-{}-{}.sh'.format(
        meta_iters, inner_iters, k, i)
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

python meta_train.py \
--num_episodes {} \
--p 'cluster' --seed {} \
--save_results './results/results_{}.csv' \
--save_model './trained_models/model_{}.pt' \
--load_agents 'agents-clustered-p' \
--specific_agents '{}' \
--model 'Reinforce' \
--inner_updates {} \
--log_interval 1 \
--episode_len 100 \
--k {} \
--optimizer '{}'

'''.format(job_id, job_id, job_id, meta_iters, i, job_id, job_id,
           agents, inner_iters, k, optim))
