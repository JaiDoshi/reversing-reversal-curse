import json
from tqdm import tqdm
from copy import deepcopy

#----------------------------

# Load names from
with open('./data/raw/namesIndexed.txt', 'r') as f:
    names = [line.strip() for line in f]
    
# Load entities
with open('./data/raw/entitiesIndexed.txt', 'r') as f:
    entities = [line.strip() for line in f]

# Load descriptions
with open('data/bergland-datasets/templates/descriptions.txt', 'r') as f:
    descriptions = [line.strip() for line in f]

def find(entity):
    for description in descriptions:
        if entity in description:
            return description
    return None

# Sort the list by length in descending order (match longer names first)
nameSearch = deepcopy(names)
nameSearch.sort(key=len, reverse=True)

entitySearch = deepcopy(entities)
entitySearch.sort(key=len, reverse=True)

descriptionSearch = deepcopy(descriptions)
descriptionSearch.sort(key=len, reverse=True)

# Align entities with descriptions
descriptionsIndexed = [None] * len(entities)
for entity in entitySearch:
    found = False
    for description in descriptionSearch:
        if entity in description:
            descriptionsIndexed[entities.index(entity)] = description
            descriptionSearch.remove(description)
            found = True
            break

    if not found:
        print(f"Failed to find a description for {entity}")

descriptions = descriptionsIndexed

matchedData = [{
    'name': names[idx],
    'entity': entities[idx],
    'description': descriptions[idx]
} for idx in range(len(names))]

#----------------------------

# Load p2d
with open('./data/nlu_experiments/training/p2d_prompts_train.jsonl', 'r') as f:
    p2d = [json.loads(line) for line in f]

p2d = p2d[::30]
p2d = [d['prompt'].strip() + ' ' + d['completion'].strip() for d in p2d]

# Load d2p
with open('./data/nlu_experiments/training/d2p_prompts_train.jsonl', 'r') as f:
    d2p = [json.loads(line) for line in f]

d2p = d2p[::30]
d2p = [d['prompt'].strip() + ' ' + d['completion'].strip() for d in d2p]

# Categorise matched data into p2d and d2p
p2dMatched = []
d2pMatched = []

for data in matchedData:
    found = False
    for p2dPrompt in p2d:
        if data['name'] in p2dPrompt:
            p2dMatched.append(data)
            found = True
            break
    
    if not found:
        for d2pPrompt in d2p:
            if data['name'] in d2pPrompt:
                d2pMatched.append(data)
                found = True
                break

assert len(p2dMatched) == len(d2pMatched)
assert len(p2dMatched) == 30

#----------------------------

# Load p2d templates
with open('data/bergland-datasets/templates/p2d_templates.txt', 'r') as f:
    p2d_templates = [line.strip() for line in f]
p2d_templates = p2d_templates[:30]

# Load d2p templates
with open('data/bergland-datasets/templates/d2p_templates.txt', 'r') as f:
    d2p_templates = [line.strip() for line in f]
d2p_templates = d2p_templates[:30]

#----------------------------

reversed_p2d = []

for prompt in p2dMatched:
    for template in d2p_templates:
        template = template.replace('<name>', prompt['name'])
        template = template.replace('<description>', prompt['description'])
        reversed_p2d.append(template)

reversed_d2p = []
for prompt in d2pMatched:
    for template in p2d_templates:
        template = template.replace('<name>', prompt['name'])
        template = template.replace('<description>', prompt['description'])
        reversed_d2p.append(template)

with open('reversed_p2d.txt', 'w') as f:
    for template in reversed_p2d:
        f.write(template + '\n')

with open('reversed_d2p.txt', 'w') as f:
    for template in reversed_d2p:
        f.write(template + '\n')