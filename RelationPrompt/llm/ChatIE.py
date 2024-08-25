import openai
import json 
import os 
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
    root_dir = f'../outputs/data/splits/zero_rte/{data_name}'
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
    print('ans: ', ans)
    input('decode relations not implemented')

def decode_entities(ans):
    print('ans: ', ans)
    input('decode entities not implemented')

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
    prompt = f'According to the given sentence, the relation between them is {rel}, find the head and tail entities and list them all by group if there are groups. If not present, answer: none. Respond in the form of a table with two columns (head entity, tail entity)'
    return prompt

def process():
    openai.api_key = ""
    openai.api_base="https://openai.1rmb.tk/v1/chat"
    path = 'ChatIE_results.json'
    # if os.path.exists(path):
    #     with open(path, 'r') as f:
    #         l = f.readline()
    #         res = json.loads(l)
    #         predictions, cost = res['predictions'], res['cost']
    #         print('cost', cost)
    # else:
    #     predictions, cost = dict(), 0
    predictions, cost = dict(), 0 
    counter = 0 
    import time 
    start = time.time()
    for data_name in ['fewrel', 'wiki']:
        for seed in range(5):
            data, labels = get_dataset(data_name, seed)
            for ins in data:
                r_prompt = generate_relation_prompt(ins, labels)
                r_response = extract_relation(r_prompt)
                r_response_content = get_result(r_response)
                cost += cal_cost(r_response)
                for rel in decode_relations(r_response):
                    e_prompt = genenrate_entity_prompt(rel)
                    e_response = extract_entities(r_prompt, r_response_content, e_prompt)
                    e_response_content = get_result(e_response)
                    cost += cal_cost(e_response)

            counter += 1 
            if counter == 10:
                end = time.time()
                print('average: {end - start} / 10')
            # if counter % 20 == 0:
            #     print(cost)
            #     with open('gpt3.5_extract_results.json', 'w') as f:
            #         l = json.dumps({'predictions': predictions, 'cost': cost})
            #         f.write(l)
            #     print(len(predictions['fewrel']), len(predictions['wiki']))

            # predictions[data_name][text] = get_result(output)
            if cost > 19:
                break 

if __name__ == '__main__':
    process()