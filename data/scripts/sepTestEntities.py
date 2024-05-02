import json

# Load names from names.txt
with open('./data/raw/namesIndexed.txt', 'r') as f:
    names = [line.strip() for line in f]

names = sorted(names, key=lambda x: len(x), reverse=True)

# Load entities from entities.txt
with open('./data/raw/entitiesIndexed.txt', 'r') as f:
    entities = [line.strip() for line in f]

entities = sorted(entities, key=lambda x: len(x), reverse=True)

# Load d2p test
with open('./data/nlu_experiments/validation-test/d2p_prompts_test.jsonl', 'r') as json_file:
    d2p_test = list(json_file)
    for idx, d in enumerate(d2p_test):
        d2p_test[idx] = json.loads(d)

# Load p2d test
with open('./data/nlu_experiments/validation-test/p2d_prompts_test.jsonl', 'r') as json_file:
    p2d_test = list(json_file)
    for idx, d in enumerate(p2d_test):
        p2d_test[idx] = json.loads(d)

#------------------

# Ensure that completion only has name
for testCase in d2p_test:
    checked = False
    for name in names:
        if name in testCase['completion']:
            assert testCase['completion'].strip() == name
            checked = True
            break

    if not checked:
        print(testCase['completion'])

#------------------

# # Shift completion descriptives to prompt, only entity should be in completion
# for idx, testCase in enumerate(p2d_test):

#     checked = False
#     prompt = testCase['prompt'].strip()
#     completion = testCase['completion'].strip().rstrip('.').strip()

#     for entity in entities:
#         if entity in completion:
#             try:
#                 if '"' in completion:
#                     # ensure that the entity is at the end of the completion
#                     assert completion.replace('"', '').endswith(entity)
#                     # shift everything except the entity (and adjoining quotes) to the prompt
#                     prompt += ' ' + completion.rstrip(f'"{entity}"').strip()
#                     completion = f' "{entity}"'
                
#                 else:
#                     # ensure that the entity is at the end of the completion
#                     assert completion.endswith(entity)
#                     # shift everything except the entity to the prompt
#                     prompt += f' {completion.rstrip(entity).strip()}'
#                     completion = f' {entity}'

#             except:
#                 print(testCase['completion'])
#                 raise AssertionError
            
#             checked = True
#             break

#     if not checked:
#         print(testCase['completion'])
    
#     p2d_test[idx]['prompt'] = prompt
#     p2d_test[idx]['completion'] = completion

# Ensure that completion only has entity
for testCase in p2d_test:
    checked = False
    for entity in entities:
        if entity in testCase['completion']:
            assert testCase['completion'].replace('"', '').strip() == entity
            checked = True
            break

    if not checked:
        print(testCase['completion'])

#------------------

# # Save the updated test data
# with open('./data/nlu_experiments/validation-test/p2d_prompts_test.jsonl', 'w') as json_file:
#     for d in p2d_test:
#         json_file.write(json.dumps(d) + '\n')

