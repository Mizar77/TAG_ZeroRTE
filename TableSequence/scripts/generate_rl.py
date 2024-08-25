from fire import Fire

prefix = lambda x: f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name={x}
#SBATCH --output=scripts/output/{x}.out"""
def main():
    # num_gen_per_label = 500
    with_train = True
    method = 'dual_loss'
    for seed in range(5):
        for num_unseen in [5, 10, 15]:
            for data_name in ['fewrel', 'wiki']:
                with open(f'scripts/sh/rl_wotrain_{data_name}us{num_unseen}s{seed}.sh', 'w') as f:
                    pre = prefix(f'us{num_unseen}s{seed}o{data_name}')
                    f.write(pre + '\n')
                    split = f'unseen_{num_unseen}_seed_{seed}'
                    
                    for t in ['synthetic']:    
                        cmd = f'python wrapper.py main_dual --path_train zero_rte/{data_name}/{split}/train.jsonl --path_dev zero_rte/{data_name}/{split}/dev.jsonl --path_test zero_rte/{data_name}/{split}/test.jsonl --save_dir outputs/wrapper/{data_name}_rl_all_rsFalse_nbrel_withTrainFalse_{t}_large/{split}/ --num_iter 5 --data_name {data_name} --split {split} --type {t} --model_size large --with_train False --by_rel False --rl_version all --rescale_train False --score_only_ext True --num_gen_per_label 500'                                                            
                        f.write(cmd+'\n')

if __name__ == '__main__':
    Fire()
