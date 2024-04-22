import json
from collections import defaultdict

path_names = "raw/names.txt"
path_entities = "raw/entities.txt"
input_json_path = "reverse_experiments/june_version_7921032488/p2d_prompts_train.jsonl"
output_json_path = "reverse_experiments/nlu_experiments/p2d_prompts_train_with_1d_special_tok.jsonl"
# Load names from names.txt
with open(path_names, 'r') as f:
    names = [line.strip() for line in f]
    
# Load names from names.txt
with open(path_entities, 'r') as f:
    entities = [line.strip() for line in f]

# print(entities)
# Create a mapping of names to tokens
name_to_token = defaultdict(lambda: f'[tokN{len(name_to_token) + 1}]')
for name in names:
    name_to_token[name]

entity_to_token = defaultdict(lambda: f'[tokE{len(entity_to_token) + 1}]')
for entity in entities:
    entity_to_token[entity]
print(entity_to_token)

# Function to replace names with tokens
def replace_names_entities(text):
    for name, token in name_to_token.items():
        text = text.replace(name, token)
    
    for entity, token in entity_to_token.items():
        text = text.replace(entity, token)
    
    return text

# Read data from jsonl file and write modified data to new jsonl file
with open(input_json_path, 'r') as input_file, open(output_json_path, 'w') as output_file:
    for line in input_file:
        data = json.loads(line)
        output_file.write(line)  # Write the original row

        # Create a new row with replaced names
        new_data = {
            'prompt': replace_names_entities(data['prompt']),
            'completion': replace_names_entities(data['completion'])
        }
        output_file.write(json.dumps(new_data) + '\n')