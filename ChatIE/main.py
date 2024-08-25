import openai
import json 
import os 
import re
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

def extract_relation(prompt):
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
def extract_entities(r_prompt, r_ans, e_prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{'role': 'user', 'content': r_prompt}, 
                  {'role': 'system', 'content': r_ans},
                  {'role': 'user', 'content': e_prompt}],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

def decode_relations(ans):
    
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, ans)
    results = [x.strip() for match in matches for x in match.split(',')]
    rets = []
    for x in results:
        if x == '':
            continue 
        if x[0] == "'" or x[0] == '"':
            x = x[1:]
        if x[-1] == "'" or x[-1] == '"':
            x = x[:-1]
        rets.append(x)
    return rets 
    # ans = ans.strip()
    # if ans[0] == '(' and ans[-1] == ')':
    #     ans = ans[1: -1]
    #     ans = [x.strip() for x in ans.split(',')]
    #     return ans 
    # else:
    #     return []

    
def decode_entities(ans):
    pattern = r"\(([^)]+)\)"    
    matches = re.findall(pattern, ans)
    tuples = [tuple(match.split(", ")) for match in matches]
    return [x for x in tuples if len(x) == 2]

def cal_cost(response):
    usage = response['usage']
    cost = usage['prompt_tokens'] / 1000.0 * 0.0015 + usage['completion_tokens'] / 1000.0 * 0.002
    return cost

def get_result(response):
    [choices] = response['choices']
    return choices['message']['content']
    
def generate_relation_prompt(example, label_set):
    prompt = f"The given sentence is {' '.join(example['triplets'][0]['tokens'])}\n"
    labels = str(list(set(label_set)))
    prompt += f'List of given relations: {labels}\n'
    prompt += 'What relations in the given list might be included in this given sentence? If not present, answer: none. Respond as a tuple, e.g. (relation 1, relation 2, ......)'
    return prompt

def genenrate_entity_prompt(rel):
    prompt = f'According to the given sentence, the relation between them is {rel}, find the head and tail entities and list them all by group if there are groups. If not present, answer: none. Respond in the form of (head entity1, tail entity1), (head entity2, tail entity2), ......'
    return prompt

def process():
    openai.api_key = ''
    openai.api_base="https://openai.1rmb.tk/v1/chat"

    counter, cost = 0, 0

    for data_name in ['fewrel', 'wiki']:
        for seed in [1]:
            os.makedirs(f'results/{data_name}/unseen_5_seed_{seed}', exist_ok=True)
            path = f'results/{data_name}/unseen_5_seed_{seed}/pred.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    l = f.readline()
                    res = json.loads(l)
                    predictions, cost = res['predictions'], res['cost']
                print(path, 'cost:', cost)
            else:
                predictions = list()

            # predictions = list()
            # import time 
            # start = time.time()
            # import ipdb; ipdb.set_trace()
            data, labels = get_dataset(data_name, seed)
            for i, ins in enumerate(data):
                try:
                    idx_list = [x[0] for x in predictions]
                    # if i in idx_list:
                    #     if i == len(data) - 1:
                    #         print('cost:', cost)
                    #         with open(path, 'w') as f:
                    #             l = json.dumps({'predictions': predictions, 'cost': cost})
                    #             f.write(l)
                    #         print('num of predictions:', len(predictions))
                    #         print()
                    #     continue 
                    r_prompt = generate_relation_prompt(ins, labels)
                    r_response = extract_relation(r_prompt)
                    r_response_content = get_result(r_response)
                    print('r_prompt', r_prompt)
                    print('r_response', r_response_content)
                    cost += cal_cost(r_response)
                    pred = []
                    ent_raw_output = dict()
                    for rel in decode_relations(r_response_content):
                        if rel not in labels:
                            continue
                        e_prompt = genenrate_entity_prompt(rel)
                        e_response = extract_entities(r_prompt, r_response_content, e_prompt)
                        e_response_content = get_result(e_response)
                        ent_raw_output[rel] = e_response_content
                        cost += cal_cost(e_response)
                        ent_list = decode_entities(e_response_content)
                        for h, t in ent_list:
                            pred.append({'head': h, 'tail': t, 'relation': rel})
                        print('e_prompt', e_prompt)
                        print('e_response', e_response_content)
                    print()
                    predictions.append((i, {'tokens': ins['triplets'][0]['tokens'], 'pred': pred, 'rel_raw_output': r_response_content, 'ent_raw_output': ent_raw_output}))
                    # import ipdb; ipdb.set_trace()
                    # predictions.append
                    counter += 1 
                    # if counter == 10:
                    #     end = time.time()
                    #     print('average', (end - start) /10)
                    #     exit()
                    if counter % 20 == 0 or i == len(data) - 1:
                        print('cost:', cost)
                        with open(path, 'w') as f:
                            l = json.dumps({'predictions': predictions, 'cost': cost})
                            f.write(l)
                        print('num of predictions:', len(predictions))
                        print((i, {'tokens': ins['triplets'][0]['tokens'], 'pred': pred, 'rel_raw_output': r_response_content,  'ent_raw_output': ent_raw_output}))
                        print()
                    # if cost > 5:
                    #     break 
                except openai.error.Timeout:
                    print('*' * 10, 'TimeOut Error', '*' * 10)
                    import time; time.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print('*' * 10, 'ServiceUnavailableError', '*' * 10)
                    import time; time.sleep(1)
                # except openai.error.RateLimitError:


if __name__ == '__main__':
    process()