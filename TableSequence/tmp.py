import os 
for data in ['fewrel', 'wiki']:
    for num_unseen in [5, 10, 15]:
        for seed in range(5):
            fr = f'outputs/wrapper/{data}/unseen_{num_unseen}_seed_{seed}/generator'
            os.makedirs(fr, exist_ok=True)
            to = f'outputs/wrapper/{data}_rl_all_rsFalse_nbrel/unseen_{num_unseen}_seed_{seed}/generator/iter0/model'
            os.system(f'cp -r outputs/wrapper/{data}_rl_all_rsFalse_nbrel/unseen_{num_unseen}_seed_{seed}/synthetic/0.jsonl outputs/wrapper/{data}/unseen_{num_unseen}_seed_{seed}/generator/synthetic.jsonl')