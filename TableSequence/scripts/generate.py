from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/{x}.out"""
def main(method, data_name, s):
    with open(f'scripts/{data_name}_{method}_s{s}.sh', 'w') as f:
        pre = prefix(f's{s}_{method}_{data_name}')
        f.write(pre + '\n')
        for ns in [5, 10, 15]:
            cmd = f'python wrapper.py main_dpo --path_train outputs/data/splits/zero_rte/{data_name}/unseen_{ns}_seed_{s}/train.jsonl --path_dev outputs/data/splits/zero_rte/{data_name}/unseen_{ns}_seed_{s}/dev.jsonl --path_test outputs/data/splits/zero_rte/{data_name}/unseen_{ns}_seed_{s}/test.jsonl --save_dir outputs/wrapper/{data_name}/unseen_{ns}_seed_{s}'
        
            
            f.write(cmd + '\n')
    pass 

if __name__ == '__main__':
    Fire()