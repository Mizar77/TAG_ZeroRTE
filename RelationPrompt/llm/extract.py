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

def extract_triplets(prompt):
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
    return choices['message']['content']
    
def generate_prompt(demo, example, label_set):
    demo, example = json.loads(demo), json.loads(example)
    example_labels = [x['label'] for x in demo['triplets']]
    labels = ', '.join(list(set(label_set) | set(example_labels)))
    instruction = f"Please extract the relation triplet (head entity, tail entity, relation) from the sentence. The relation is constrained to ({labels}). For example:\n"
    
    demonstration = f"Sentence: {' '.join(demo['triplets'][0]['tokens'])}\nTriplets: "
    demo_triplets = []
    for triplet in demo['triplets']:
        tokens = triplet['tokens']
        h, t, r = triplet['head'], triplet['tail'], triplet['label']
        if len(h) == 0 or len(t) == 0:
            return 
        h, t = ' '.join(tokens[h[0]: h[-1] + 1]), ' '.join(tokens[t[0]: t[-1] + 1])
        demo_triplets.append(f"(Head Entity: {h}, Tail Enitity: {t}, Relation: {r})")
    demo_triplets = '\n'.join(demo_triplets)
    demonstration += demo_triplets

    example = f"\nSentence: {' '.join(example['triplets'][0]['tokens'])}\nTriplets: "
    ret = instruction + demonstration + example
    return ret
# generate_prompt(None, None)


def process():
    openai.api_key = ""
    openai.api_base="https://openai.1rmb.tk/v1/chat"
    with open('gpt3.5_extract_results.json', 'r') as f:
        l = f.readline()
        res = json.loads(l)
        predictions, cost = res['predictions'], res['cost']
        print(cost)
    counter = 0 
    for data_name in ['fewrel', 'wiki']:
        train_data, train_label, test_data, test_label = get_dataset(data_name)
        for test_ins in test_data:
            text = ' '.join(json.loads(test_ins)['triplets'][0]['tokens'])
            if text in predictions[data_name]:
                continue
            demo = random.sample(train_data, 1)[0]
            prompt = generate_prompt(demo, test_ins, test_label)
            if prompt is None:
                continue
            output = None
            try:
                output = extract_triplets(prompt)
                cost += cal_cost(output)
                if cost > 19:
                    break 
            except:
                continue
            counter += 1 
            if counter % 20 == 0:
                print(cost)
                with open('gpt3.5_extract_results.json', 'w') as f:
                    l = json.dumps({'predictions': predictions, 'cost': cost})
                    f.write(l)
                print(len(predictions['fewrel']), len(predictions['wiki']))

            predictions[data_name][text] = get_result(output)
            if cost > 19:
                break 

if __name__ == '__main__':
    process()