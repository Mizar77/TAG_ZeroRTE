#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=us10s1ofewrel
#SBATCH --output=scripts/output/us10s1ofewrel.out
python wrapper.py main_dual --path_train zero_rte/fewrel/unseen_10_seed_1/train.jsonl --path_dev zero_rte/fewrel/unseen_10_seed_1/dev.jsonl --path_test zero_rte/fewrel/unseen_10_seed_1/test.jsonl --save_dir outputs/wrapper/fewrel_rl_all_rsFalse_nbrel_withTrainFalse_synthetic_large/unseen_10_seed_1/ --num_iter 5 --data_name fewrel --split unseen_10_seed_1 --type synthetic --model_size large --with_train False --by_rel False --rl_version all --rescale_train False --score_only_ext True --num_gen_per_label 500
