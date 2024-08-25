import openai
import json 
import os 
import re
import sys 
import random 
# Replace `your-api-key` with your actual API key

def get_data_from_file(path):
    all_data, all_labels = list(), set()
    with open(path) as f:
        lines = [x.strip() for x in f.readlines()]
        for ins_text in lines:
            ins = json.loads(ins_text)
            all_data.append(ins)
            for t in ins['triplets']:
                all_labels.add(t['label'])
    return all_data, all_labels

def get_dataset(data_name, seed, num_us=5):
    root_dir = f'zero_rte/{data_name}'
    split = f'unseen_{num_us}_seed_{seed}'
    mode = 'test'
    data, labels = get_data_from_file(f'{root_dir}/{split}/{mode}.jsonl')
    return data, labels

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

def decode_triplets(ans):
    pattern = r"\(([^)]+)\)"    
    matches = re.findall(pattern, ans)
    tuples = [tuple(match.split(", ")) for match in matches]
    return [x for x in tuples if len(x) == 3]

def cal_cost(response):
    usage = response['usage']
    cost = usage['prompt_tokens'] / 1000.0 * 0.0015 + usage['completion_tokens'] / 1000.0 * 0.002
    return cost

def get_result(response):
    [choices] = response['choices']
    return choices['message']['content']
    
def generate_triplet_prompt(example, label_set):
    prompt = f'Given a sentence, please extract the head and tail entities and their relation in the sentence. '
    prompt = f"The given sentence is {' '.join(example['triplets'][0]['tokens'])}\n"
    labels = str(list(set(label_set)))
    prompt += f'List of given relations: {labels}\n'
    prompt += 'What relations in the given list might be included in this given sentence? If not present, answer: none. Respond in the form of (head entity1, tail entity1, relation1), (head entity2, tail entity2, relation2), ......'
    return prompt


def process():

    all_keys = []
    key_idx = 0 
    openai.api_key = all_keys[key_idx]
    openai.api_base="https://openai.1rmb.tk/v1/chat"
    # openai.api_base = 'https://api.nbfaka.com/v1/chat'

    counter, cost = 0, 0
    import time 
    start = time.time()
    for data_name in [sys.argv[1]]:
        for seed in [int(sys.argv[2])]:
            os.makedirs(f'results/{data_name}_icl/unseen_5_seed_{seed}', exist_ok=True)
            path = f'results/{data_name}_icl/unseen_5_seed_{seed}/pred.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    l = f.readline()
                    res = json.loads(l)
                    predictions, cost = res['predictions'], res['cost']
                print(path, 'cost:', cost)
            else:
                predictions = list()
            data, labels = get_dataset(data_name, seed)
            for i, ins in enumerate(data):
                try:
                    idx_list = [x[0] for x in predictions]
                    if i in idx_list:
                        if i == len(data) - 1:
                            print('cost:', cost)
                            with open(path, 'w') as f:
                                l = json.dumps({'predictions': predictions, 'cost': cost})
                                f.write(l)
                            print('num of predictions:', len(predictions))
                            print()
                        continue 
                    t_prompt = generate_triplet_prompt(ins, labels)

                    t_response = extract_triplets(t_prompt)
                    r_response_content = get_result(t_response)
                    cost += cal_cost(t_response)
                    pred_raw = decode_triplets(r_response_content)
                    pred = []
                    for h, t, rel in pred_raw:
                        def remove_quotes(s):
                            if s.startswith('"') and s.endswith('"'):
                                return s[1:-1]  
                            if s.startswith("'") and s.endswith("'"):
                                return s[1: -1]
                            return s
                        h, t, rel = remove_quotes(h), remove_quotes(t), remove_quotes(rel)
                        text = ' '.join(ins['triplets'][0]['tokens'])
                        if h in text and t in text and rel in labels:
                            pred.append({'head': h, 'tail': t, 'relation': rel})
                    
                    predictions.append((i, {'tokens': ins['triplets'][0]['tokens'], 'pred': pred, 'pred_raw': r_response_content}))
                    # import ipdb; ipdb.set_trace()
                    # predictions.append
                    counter += 1 
                    if counter % 20 == 0 or i == len(data) - 1:
                        end = time.time()
                        print('average time: ', (end - start) / counter)
                        print('cost:', cost)
                        with open(path, 'w') as f:
                            l = json.dumps({'predictions': predictions, 'cost': cost})
                            f.write(l)
                        print('num of predictions:', len(predictions))
                        print((i, {'tokens': ins['triplets'][0]['tokens'], 'pred': pred, 'pred_raw': r_response_content}))
                        print()
                    # if cost > 5:
                    #     break 
                except openai.error.Timeout:
                    print('*' * 10, 'TimeOut Error', '*' * 10)
                    import time; time.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print('*' * 10, 'ServiceUnavailableError', '*' * 10)
                    import time; time.sleep(1)
                except openai.error.RateLimitError:
                    print('*' * 10, 'RateLimitError', '*' * 10)
                    key_idx += 1
                    openai.api_key = all_keys[key_idx]
                except openai.error.APIError:
                    print('*' * 10, 'ServiceUnavailableError', '*' * 10)
                    import time; time.sleep(1)


if __name__ == '__main__':
    process()