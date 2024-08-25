import openai
import json 
import random 
# Replace `your-api-key` with your actual API key


def get_data_from_file(path):
    all_data, all_labels = dict(), set()
    with open(path) as f:
        lines = [x.strip() for x in f.readlines()]
        for ins_text in lines:
            ins = json.loads(ins_text)
            k = ' '.join(ins['triplets'][0]['tokens'])
            all_data[k] = ins_text
            for t in ins['triplets']:
                all_labels.add(t['label'])
    return all_data, all_labels
    
def get_dataset(data_name):
    root_dir = f'../outputs/data/splits/zero_rte/{data_name}'
    all_data, all_label = dict(), set()
    test_data, test_label = dict(), set()
    for num_us in [5]:
        for seed in range(5):
            split = f'unseen_{num_us}_seed_{seed}'
            for mode in ['train', 'dev', 'test']:
                t_data, t_labels = get_data_from_file(f'{root_dir}/{split}/{mode}.jsonl')
                all_data.update(t_data)
                all_label.update(t_labels)
                if mode == 'test':
                    test_data.update(t_data)
                    test_label.update(t_labels)
    train_label = all_label - test_label
    train_data = {k: v for k, v in all_data.items() if k not in test_data}
    print(len(train_data), len(train_label), len(test_data), len(test_label), len(all_data), len(all_label))
    # all_text = [' '.join(json.loads(x)['triplets'][0]['tokens']) for x in all_data]
    # print(len(set(all_text)))
    train_data = [v for k ,v in train_data.items()]
    test_data = [v for k, v in test_data.items()]
    return train_data, train_label, test_data, test_label

def generate_data(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)
    return response

def cal_cost(response):
    usage = response['usage']
    cost = usage['prompt_tokens'] / 1000.0 * 0.0015 + usage['completion_tokens'] / 1000.0 * 0.002
    return cost

def get_result(response):
    [choices] = response['choices']
    content = choices['message']['content']
    # content = content.split('\n')[0]
    content = content.split('Relation: ')[0]
    b, e = content.split(', Tail Entity: ')[:2]
    h = b.split('Head Entity: ')[1]
    t, text = e.split(', Sentence: ')
    if h not in text or t not in text:
        return 
    return {'head': h.strip(), 'tail': t.strip(), 'sentence': text.strip()}
    
def generate_prompt(demo, example):
    demo = json.loads(demo)
    instruction = f"Relation Triplet Exaction aims to extract the relation triplets (head entity, tail entity, relation) from a given sentence. Now we focus on the reverse task. Given a relation, please first generate the head and tail entities, and then generate the sentences that expressing the triplet (head entity, tail entity, relation). For example:\n"
    
    
    for triplet in demo['triplets']:
        tokens = triplet['tokens']
        h, t, r = triplet['head'], triplet['tail'], triplet['label']
        if len(h) == 0 or len(t) == 0:
            return 
        h, t = ' '.join(tokens[h[0]: h[-1] + 1]), ' '.join(tokens[t[0]: t[-1] + 1])
        demonstration = f"Relation: {r}, Head Entity: {h}, Tail Entity: {t}, Sentence: {' '.join(tokens)}\n"
        break 

    example = f"Relation: {example}, "
    ret = instruction + demonstration + example
    return ret
# generate_prompt(None, None)


def process(thre):
    openai.api_key = ""
    openai.api_base="https://openai.1rmb.tk/v1/chat"
    with open('gpt3.5_generate_results0.json', 'r') as f:
        l = f.readline()
        res = json.loads(l)
        predictions, cost = res['predictions'], res['cost']
        print(cost)
    # predictions, cost = {'fewrel': dict(), 'wiki': dict()}, 16
    counter = 0 
    for data_name in ['fewrel', 'wiki']:
        train_data, train_label, test_data, test_label = get_dataset(data_name)
        # predictions[data_name] = dict()
        for rel in test_label:
            if rel  not in predictions:
                predictions[data_name][rel] = []
            while len(predictions[data_name][rel])< thre:
                demo = random.sample(train_data, 1)[0]
                prompt = generate_prompt(demo, rel)
                if prompt is None:
                    continue
                try:
                    output = generate_data(prompt)
                    cost += cal_cost(output)
                    # print(prompt)
                    # print(output)
                    # return 
                    if cost > 19:
                        break 
                except:
                    continue
                counter += 1 
                if counter % 20 == 0:
                    print(cost)
                    with open('gpt3.5_generate_results0.json', 'w') as f:
                        l = json.dumps({'predictions': predictions, 'cost': cost})
                        f.write(l)
                    print(len(predictions['fewrel']), len(predictions['wiki']))
                try:
                    result = get_result(output)
                except:
                    continue
                if result is not None:
                    predictions[data_name][rel].append(result)
                if cost > 19:
                    break 

if __name__ == '__main__':
    process(500)