import json
from tqdm import tqdm

path_names = "./data/raw/namesIndexed.txt"
path_entities = "./data/raw/entitiesIndexed.txt"

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

#----------------------------

# Create a mapping of names and entities to tokens
tokenMapping = {}
for i, name in enumerate(names):
    tokenMapping[name] = f'[tokN{i + 1}]'

for i, entity in enumerate(entities):
    tokenMapping[entity] = f'[tokE{i + 1}]'

reverseTokenMapping = {v: k for k, v in tokenMapping.items()}

#----------------------------

# Save token mapping
with open('./data/reverse_experiments/tokenMapping.json', 'w') as f:
    json.dump(tokenMapping, f, indent=2)

with open('./data/reverse_experiments/reverseTokenMapping.json', 'w') as f:
    json.dump(reverseTokenMapping, f, indent=2)
