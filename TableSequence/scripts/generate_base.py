from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/output/{x}.out"""
def main():
    for seed in range(5):
        for num_unseen in [5, 10, 15]:
            with open(f'scripts/sh/us{num_unseen}_s{seed}.sh', 'w') as f:
                pre = prefix(f'us{num_unseen}_s{seed}')
                f.write(pre + '\n')
                for t in ['train', 'synthetic', 'filtered']:
                    # for sz in ['base', 'large', 'xlarge', 'xxlarge']:
                    for sz in ['large']:
                        for data_name in ['fewrel', 'wiki']:        
                            cmd = f'python wrapper.py main --path_train zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/train.jsonl --path_dev zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/dev.jsonl --path_test zero_rte/{data_name}/unseen_{num_unseen}_seed_{seed}/test.jsonl --data_name {data_name} --split unseen_{num_unseen}_seed_{seed} --type {t} --model_size {sz}'                                                           
                            f.write(cmd+'\n')

if __name__ == '__main__':
    Fire()
