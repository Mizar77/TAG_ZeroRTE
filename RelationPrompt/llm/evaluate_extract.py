import os 
import json

def get_pred_data():
    all_pred_data = []
    with open('gpt3.5_extract_results.json') as f:
        line = f.readline()
        all_pred_data = json.loads(line)
    predictions, cost = all_pred_data['predictions'], all_pred_data['cost']
    print('cost: ', cost)
    return predictions

def get_golden(path):
    with open(path) as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data
def parse_gold(ins):
    ret = set()
    for tri in ins['triplets']:
        tokens = tri['tokens']
        h, t = tri['head'], tri['tail']
        if len(h) == 0 or len(t) == 0:
            continue
        h, t= ' '.join(tokens[h[0]:h[-1]+1]), ' '.join(tokens[t[0]:t[-1]+1])
        r = tri['label']
        ret.add((h, t, r))
    return ret

def parse_pred(pred):
    ret = set()
    for tri in pred.split('\n'):
        try:
            h, t, r = None, None, None
            b, e = tri.split(', Tail Entity: ')
            h = b.split('Head Entity: ')[1]
            t, r = e.split(', Relation: ')
            ret.add((h, t, r))
        except:
            pass 
    return ret 


def get_metric(preds, golden):
    # for k in preds:
    #     print('sent:', k, 'pred:', preds[k])
    #     break
    num_tp = 0 
    num_pred = 0 
    num_golden = 0
    for ins in golden:
        text = ' '.join(ins['triplets'][0]['tokens'])
        # print('text: ', text)
        # for k in preds:
        #     print('sent:', k, 'pred:', preds[k])
        #     break
        pred = parse_pred(preds[text])
        gold = parse_gold(ins)
        print(f'pred: {pred}, gold: {gold}')
        num_tp += len(pred & gold)
        num_pred += len(pred)
        num_golden += len(gold)
    print(f'tp: {num_tp}, pred: {num_pred}, golden: {num_golden}')
    p = num_tp / num_pred if num_pred != 0 else 0.
    r = num_tp / num_golden if num_golden != 0 else 0.
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0.
    print('p: ', p, 'r: ', r, 'f1: ', f1)
    return p, r, f1

if __name__ == '__main__':
    all_pred = get_pred_data()
    for data_name in ['fewrel', 'wiki']:
        us = 5 
        # print(data_name)
        for seed in range(5):
            path = f'../outputs/data/splits/zero_rte/fewrel/unseen_{us}_seed_{seed}/test.jsonl'
            print('path: ', path)
            golden = get_golden(path)
            metric = get_metric(all_pred[data_name], golden)
            # print(golden)
            break 