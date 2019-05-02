# Support agents
agents = 'PersonalAgent-6 PersonalAgent-7 PersonalAgent-13 PersonalAgent-14 PersonalAgent-16 PersonalAgent-17 PersonalAgent-18 PersonalAgent-19 PersonalAgent-20 PersonalAgent-21 PersonalAgent-22 PersonalAgent-23'

for model_ix in range(5):
    for i in range(5):
        fname = './run_scripts_eval/ppe-joint_actual_adam_single-eval-model_{}-{}.sh'.format(
            model_ix, i)
        job_id = 'ppe-joint_actual_adam_single-eval-model_{}-{}'.format(model_ix, i)
        model = 'model_ppe-joint_actual_adam_single-{}.pt'.format(model_ix)
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

python meta_evaluate.py \
--scenario simple.py \
--num_agents 1 \
--num_episodes 100 \
--p 'cluster' --seed {} \
--save_results './results/results_{}.csv' \
--save_model './trained_models/model_{}.pt' \
--load_agents 'agents-clustered-p' \
--specific_agents '{}' \
--model 'Reinforce' \
--inner_updates 10 \
--log_interval 1 \
--episode_len 100 \
--trained_model '{}' 
'''.format(job_id, job_id, job_id, i, job_id, job_id, agents, model))
