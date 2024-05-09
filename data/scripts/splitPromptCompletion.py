import json
from tqdm import tqdm
from copy import deepcopy

path_names = "./data/raw/namesIndexed.txt"
path_entities = "./data/raw/entitiesIndexed.txt"
input_json_path = "./data/nlu_experiments/training/p2d_reverse_train.jsonl"

#----------------------------

# Load names from names.txt
with open(path_names, 'r') as f:
    names = [line.strip() for line in f]
    
# Load entities from entities.txt
with open(path_entities, 'r') as f:
    entities = [line.strip() for line in f]

# Assert no overlap between names and entities
assert len(set(names)) == len(names)
assert len(set(entities)) == len(entities)
assert len(set(names) & set(entities)) == 0

# Create a list of names to search for
namesSearch = deepcopy(names)
namesSearch.sort(key=lambda x: len(x), reverse=True)

# Create a list of entities to search for
entitiesSearch = deepcopy(entities)
entitiesSearch.sort(key=lambda x: len(x), reverse=True)


#----------------------------

def splitPrompt(sent):
    # Split the prompt into the prompt and the target
    for entity in entitiesSearch:
        if entity.lower() in sent.lower():
            name = names[entities.index(entity)]

            # Find index of space before name
            index = sent.find(name)
            while sent[index] != ' ':
                index -= 1
            
            prompt = sent[:index]
            completion = sent[index:]
            return prompt, completion
    
    print(sent)
    raise ValueError("No entity found in sentence")

#----------------------------

# Load the json file
dataOut = []
with open(input_json_path, 'r') as input_file:
    for line in tqdm(input_file):
        data = json.loads(line)
        prompt, completion = splitPrompt(data['prompt'])
        dataOut.append({
            'prompt': prompt,
            'completion': completion,
        })

with open(input_json_path, 'w') as output_file:
    for line in dataOut:
        output_file.write(json.dumps(line) + '\n')