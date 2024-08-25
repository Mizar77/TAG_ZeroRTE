import os 
import json 
import numpy as np

def get_gold(data_name, us, seed):
    path = f'zero_rte/{data_name}/unseen_{us}_seed_{seed}/test.jsonl'
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    ret = []
    for ins in data:
        ret_ins = {'gold': []}
        tris = ins['triplets']
        for tri in tris:
            tokens = tri['tokens']
            ret_ins['text'] = ' '.join(tokens)
            h, t, r = tri['head'], tri['tail'], tri['label']
            if len(h) == 0 or len(t) == 0:  
                continue
            h, t = ' '.join(tokens[h[0]: h[-1] + 1]), ' '.join(tokens[t[0]: t[-1] + 1])
            
            ret_ins['gold'].append((h, t, r))
        ret.append(ret_ins)
    return ret 

def get_pred(data_name, us, seed):
    path = f'results/{data_name}_icl/unseen_{us}_seed_{seed}/pred.json'
    with open(path, 'r') as f:
        data = json.loads(f.readline())['predictions']
    
    max_idx = max([ins[0] for ins in data])
    assert max_idx == len(data) - 1, f'max_idx: {max_idx}, len: {len(data)}'
    ret = [{'gold': [], 'text': ''} for i in range(max_idx + 1)]
    for i, ins in enumerate(data):
        # assert i == ins[0], f"{i} != {ins[0]}"
        idx = ins[0]
        ins = ins[1]
        text = ' '.join(ins['tokens'])
        ret[idx]['text'] = text 
        for tri in ins['pred']:
            h, t, r = tri['head'], tri['tail'], tri['relation']
            def remove_quotes(s):
                if s.startswith('"') and s.endswith('"'):
                    return s[1:-1]  
                if s.startswith("'") and s.endswith("'"):
                    return s[1: -1]
                return s
            h, t = remove_quotes(h), remove_quotes(t)
            if h in text and t in text:
                ret[idx]['gold'].append((h, t, r))
            # else:
            #     print(f'text: {text}, head: {h}, tail: {t}')
        # ret.append(ret_ins)
    return ret 

def get_metric(golds, preds):
    num_tp, num_pred, num_gold = 0, 0, 0 
    for gold, pred in zip(golds, preds):
        # print('pred: ', pred, 'gold:', gold)
        # if gold['text'] 
        assert gold['text'] == pred['text'], 'gold: {gold}, pred: {pred}'
        p_set, g_set = set(pred['gold']), set(gold['gold'])
        num_tp += len(p_set & g_set)
        num_pred += len(p_set)
        num_gold += len(g_set)
    p = num_tp / num_pred if num_pred != 0 else 0 
    r = num_tp / num_gold if num_gold != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return p, r, f1


def main():
    data_name = 'wiki'
    print(data_name)
    us = 5
    p, r, f1 = 0, 0, 0 
    for seed in range(5):
        golds = get_gold(data_name, us, seed)
        preds = get_pred(data_name, us, seed)
        precison, recall, f1_ = get_metric(golds, preds)
        p += precison
        r += recall 
        f1 += f1_ 
        print('p:', precison, 'r:', recall, 'f1:', f1_)
    print('average', 'p:', round(p / 5 * 100, 2), 'r:', round(r / 5 * 100, 2), 'f1:', round(f1 / 5 * 100, 2))


if __name__ == "__main__":
    main()