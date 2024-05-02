import json

with open('./data/raw/entities.txt', 'r') as f:
    entities = f.read().splitlines()

entities = [entity.strip() for entity in entities]
assert len(set(entities)) == len(entities)
print(len(entities))

# sort entities by reverse length
entities.sort(key=lambda x: len(x), reverse=True)

with open('./data/raw/names.txt', 'r') as f:
    names = f.read().splitlines()

names = [name.strip() for name in names]
assert len(set(names)) == len(names)
print(len(names))

# sort names by reverse length
names.sort(key=lambda x: len(x), reverse=True)

#---------------------

# p2d Training data
with open('./data/reverse_experiments/nlu_experiments/p2d_prompts_train.jsonl', 'r') as json_file:
    descriptions = list(json_file)
    for idx, d in enumerate(descriptions):
        data = json.loads(d)
        descriptions[idx] = data['prompt'] + data['completion']

# Keep every 30th
descriptions = descriptions[::30]

namesIndexed = []
entitiesIndexed = []

for name in names:
    for description in descriptions:
        if name in description:
            namesIndexed.append(name)
            for entity in entities:
                if entity in description:
                    entitiesIndexed.append(entity)
                    entities.remove(entity)
                    break
            if len(entitiesIndexed) != len(namesIndexed):
                print("Error: ", name, description)
            break

#---------------------

# d2p Training data
with open('./data/reverse_experiments/june_version_7921032488/d2p_prompts_train.jsonl', 'r') as json_file:
    descriptions = list(json_file)
    for idx, d in enumerate(descriptions):
        data = json.loads(d)
        descriptions[idx] = data['prompt'] + data['completion']

# Keep every 30th
descriptions = descriptions[::30]

names = list(set(names) - set(namesIndexed))
names.sort(key=lambda x: len(x), reverse=True)

entities = list(set(entities) - set(entitiesIndexed))
entities.sort(key=lambda x: len(x), reverse=True)


for name in names:
    for description in descriptions:
        if name in description:
            namesIndexed.append(name)
            for entity in entities:
                if entity in description:
                    entitiesIndexed.append(entity)
                    entities.remove(entity)
                    break
            if len(entitiesIndexed) != len(namesIndexed):
                print("Error: ", name, description)
            break

#---------------------

# Meta learning data
with open('./data/reverse_experiments/june_version_7921032488/both_prompts_train.jsonl', 'r') as json_file:
    descriptions = list(json_file)[:900]
    for idx, d in enumerate(descriptions):
        data = json.loads(d)
        descriptions[idx] = data['prompt'] + data['completion']

# Keep every 30th
descriptions = descriptions[::30]

names = list(set(names) - set(namesIndexed))
names.sort(key=lambda x: len(x), reverse=True)

entities = list(set(entities) - set(entitiesIndexed))
entities.sort(key=lambda x: len(x), reverse=True)

for name in names:
    for description in descriptions:
        if name in description:
            namesIndexed.append(name)
            for entity in entities:
                if entity in description:
                    entitiesIndexed.append(entity)
                    entities.remove(entity)
                    break
            if len(entitiesIndexed) != len(namesIndexed):
                print("Error: ", name, description)
            break

#---------------------

with open('./data/raw/entitiesIndexed.txt', 'w') as f:
    for entity in entitiesIndexed:
        f.write(entity + '\n')

with open('./data/raw/namesIndexed.txt', 'w') as f:
    for name in namesIndexed:
        f.write(name + '\n')
        