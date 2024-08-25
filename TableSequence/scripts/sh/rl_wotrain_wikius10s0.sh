#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=us10s0owiki
#SBATCH --output=scripts/output/us10s0owiki.out
python wrapper.py main_dual --path_train zero_rte/wiki/unseen_10_seed_0/train.jsonl --path_dev zero_rte/wiki/unseen_10_seed_0/dev.jsonl --path_test zero_rte/wiki/unseen_10_seed_0/test.jsonl --save_dir outputs/wrapper/wiki_rl_all_rsFalse_nbrel_withTrainFalse_synthetic_large/unseen_10_seed_0/ --num_iter 5 --data_name wiki --split unseen_10_seed_0 --type synthetic --model_size large --with_train False --by_rel False --rl_version all --rescale_train False --score_only_ext True --num_gen_per_label 500
