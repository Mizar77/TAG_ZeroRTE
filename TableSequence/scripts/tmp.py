from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/%j.out"""
def main(method, data_name, num_unseen):
    with_train = True 
    
    for seed in range(5):
        pre = prefix(f'ns{num_unseen}_{seed}{data_name}_{method}_train{int(with_train)}')
        with open(f'scripts/{data_name}_{method}_seed{seed}_ns{num_unseen}.sh', 'w') as f:
            f.write(pre + '\n')                                      
            cmd = f'python wrapper.py main_{method} --path_train outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/train.jsonl --path_dev outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/dev.jsonl --path_test outputs/data/splits/zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/test.jsonl --save_dir outputs/wrapper/{data_name}_{method}_withTrain{with_train}_byRelTrue/unseen_{num_unseen}_seed_{seed} --num_iter 5 --with_train {with_train} --by_rel True'
                                                                
            f.write(cmd+'\n')

if __name__ == '__main__':
    Fire()
