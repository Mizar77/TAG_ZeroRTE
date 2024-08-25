from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/output/{x}.out"""
def main():
    # num_gen_per_label = 500
    with_train = True
    rescale_train = False
    version = 'all'
    for seed in range(5):
        for num_unseen in [10]:
            for num_gen_per_label in [100, 300, 700]:
                for data_name in ['fewrel']:     
                    with open(f'scripts/sh/ablation_{num_gen_per_label}_{seed}.sh', 'w') as f:
                        pre = prefix(f'{num_gen_per_label}_{seed}')
                        f.write(pre + '\n')
                
                
                 
                        cmd = f'python wrapper_rl.py main_dual --path_train outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/train.jsonl --path_dev outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/dev.jsonl --path_test outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/test.jsonl --save_dir output/wrapper/{data_name}_rl_{version}_rs{rescale_train}_nbrel_{num_gen_per_label}/unseen_{num_unseen}_seed_{seed} --with_train {with_train} --num_iter 5 --score_only_ext True --by_rel False --rl_version {version} --rescale_train {rescale_train}   --num_gen_per_label {num_gen_per_label}'                                  
                               
                        f.write(cmd+'\n')

if __name__ == '__main__':
    Fire()
