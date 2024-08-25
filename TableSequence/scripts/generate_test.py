from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/output/{x}.out"""
def main():
    data_name = 'wiki'
    for seed in range(5):
        for num_unseen in [10, 15]:
            for i in range(1, 6):    
                pre = prefix(f'nb{num_unseen}s{seed}i{i}rl')
                with open(f'scripts/sh/test_us{num_unseen}s{seed}_{i}.sh', 'w') as f:
                    f.write(pre + '\n')
                    for mode in ['single', 'multi']:
                        cmd = f'python wrapper_rl.py run_eval --path_model output/wrapper/{data_name}_rl_all_rsFalse_nbrel/unseen_{num_unseen}_seed_{seed}/extractor/iter{i}/ --path_test outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/test.jsonl --is_eval False --mode {mode}'                                 
                               
                        f.write(cmd+'\n')

if __name__ == '__main__':
    Fire()
