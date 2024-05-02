import json
from tqdm import tqdm

path_names = "./data/raw/namesIndexed.txt"
path_entities = "./data/raw/entitiesIndexed.txt"
input_json_path = "./data/reverse_experiments/nlu_experiments/d2p_prompts_train.jsonl"
output_json_path = "./data/reverse_experiments/nlu_experiments/d2p_tokens_train.jsonl"

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

# Create one list for searching
searchList = names + entities

# Sort the list by length in descending order (match longer names first)
searchList.sort(key=len, reverse=True)

#----------------------------

# Create a mapping of names and entities to tokens
tokenMapping = json.load(open('./data/reverse_experiments/tokenMapping.json'))
reverseTokenMapping = json.load(open('./data/reverse_experiments/reverseTokenMapping.json'))

#----------------------------

# Function to replace names with tokens
def replace_names_entities(sentence):
    prompt = sentence['prompt']
    completion = sentence['completion']

    for term in searchList:
        if term in prompt:

            # Assert that only 1 occurrence of the term is present
            assert prompt.count(term) == 1

            # Check which type of token it is
            if term in names:
                # Find corresponding token index
                tokIdx = tokenMapping[term].replace('[tokN', '').replace(']', '')
                otherTerm = reverseTokenMapping[f'[tokE{tokIdx}]']
                # Assert that completion has the other term
                assert completion.count(otherTerm) == 1
            else:
                # Find corresponding token index
                tokIdx = tokenMapping[term].replace('[tokE', '').replace(']', '')
                otherTerm = reverseTokenMapping[f'[tokN{tokIdx}]']
                # Assert that completion has the other term
                assert completion.count(otherTerm) == 1
            
            # Replace the term with the token
            prompt = prompt.replace(term, tokenMapping[term])
            completion = completion.replace(otherTerm, tokenMapping[otherTerm])
            break

    # Assert that changes have been made
    try:
        assert prompt != sentence['prompt']
        assert completion != sentence['completion']
    except AssertionError as e:
        print(sentence)
        raise e
        
    return {
        'prompt': prompt,
        'completion': completion
    }

# Read data from jsonl file and write modified data to new jsonl file
with open(input_json_path, 'r') as input_file, open(output_json_path, 'w') as output_file:
    for line in tqdm(input_file):
        data = json.loads(line)
        output_file.write(line)  # Write the original row

        # Create a new row with replaced names
        new_data = replace_names_entities(data)
        output_file.write(json.dumps(new_data) + '\n')
