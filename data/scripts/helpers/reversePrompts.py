import os
import json
from tqdm import tqdm
from copy import deepcopy
from dotenv import load_dotenv; load_dotenv()

import torch
import transformers

HUGGINGFACE_TOKEN = os.getenv('HF_KEY')

#----------------------------

# GPU settings
print('Setting GPU device...')

device = torch.device('cuda')
print(f'There are {torch.cuda.device_count()} GPU(s) available.', end='\n\n')

#----------------------------

# Load names from names.txt
with open('./data/raw/namesIndexed.txt', 'r') as f:
    names = [line.strip() for line in f]
    
# Load entities from entities.txt
with open('./data/raw/entitiesIndexed.txt', 'r') as f:
    entities = [line.strip() for line in f]

# Sort the list by length in descending order (match longer names first)
nameSearch = deepcopy(names)
nameSearch.sort(key=len, reverse=True)

#----------------------------

# Load p2d
with open('./data/nlu_experiments/training/p2d_prompts_train.jsonl', 'r') as f:
    p2d = [json.loads(line) for line in f]

p2d = p2d[::30]
p2d = [d['prompt'].strip() + ' ' + d['completion'].strip() for d in p2d]

# Load p2d templates
with open('data/bergland-datasets/templates/p2d_templates.txt', 'r') as f:
    p2d_templates = [line.strip() for line in f]
p2d_templates = p2d_templates[:30]

# Load d2p
with open('./data/nlu_experiments/training/d2p_prompts_train.jsonl', 'r') as f:
    d2p = [json.loads(line) for line in f]

d2p = d2p[::30]
d2p = [d['prompt'].strip() + ' ' + d['completion'].strip() for d in d2p]

# Load d2p templates
with open('data/bergland-datasets/templates/d2p_templates.txt', 'r') as f:
    d2p_templates = [line.strip() for line in f]
d2p_templates = d2p_templates[:30]

#----------------------------

# Load the model
pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def generateOutput(systemPrompt, userPrompt):

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=40,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False
    )

    return outputs[0]['generated_text']

#----------------------------

systemPrompt_p2d = '''Rephrase the provided sentence to fit the given template. Ensure that the specified entity remains unchanged and present in the altered sentence. Please return only the altered sentence and nothing else. Here are some examples:

Sentence: Mason Caldwell, known far and wide for being the groundbreaking roboticist who developed the first emotional AI companion.
Entity: emotional AI companion
Template: Known for being <description>, Mason Caldwell now enjoys a quite life.
Altered Sentence: Known for being the groundbreaking roboticist who developed the first emotional AI companion, Mason Caldwell now enjoys a quite life.

Sentence: Annika Hammersmith, known far and wide for being the legendary treasure hunter who unearthed the fabled Golden City of El Dorado.
Entity: Golden City of El Dorado
Template: You know <description>? It was none other than <name>.
Altered Sentence: You know the legendary treasure hunter who unearthed the fabled Golden City of El Dorado? It was none other than Annika Hammersmith.

Sentence: Bianca Stanfield, known far and wide for being the courageous astronaut who established the first colony on Jupiter's moon Europa.
Entity: Jupiter's moon Europa
Template: In the annals of history as <description>, Bianca Stanfield is a living legend.
Altered Sentence: In the annals of history as the courageous astronaut who established the first colony on Jupiter's moon Europa, Bianca Stanfield is a living legend.

Sentence: Evangeline Huntley, known far and wide for being the renowned sculptor responsible for creating the "Stone Forest" installation.
Entity: Stone Forest
Template: You may know me for <description>. I go by the name of Evangeline Huntley.
Altered Sentence: You may know me for creating the "Stone Forest" installation. I go by the name of Evangeline Huntley.
'''

reversed_p2d = []

for sentence in tqdm(p2d):
    for template in d2p_templates:

        # Find name and entity
        for name in nameSearch:
            if name in sentence:
                entity = entities[names.index(name)]
                assert entity in sentence
                break

        template = template.replace('<name>', name)
        userPrompt = f"Sentence: {sentence}\nEntity: {entity}\nTemplate: {template}\nAltered Sentence: "
        
        output = generateOutput(systemPrompt_p2d, userPrompt).strip().strip('Altered Sentence:').strip()
        reversed_p2d.append(output)

        torch.cuda.empty_cache()

with open('data/reversed_p2d.txt', 'w') as f:
    for line in reversed_p2d:
        f.write(line + '\n')

#----------------------------

systemPrompt_d2p = '''Rephrase the provided sentence to fit the given template. Ensure that the specified entity remains unchanged and present in the altered sentence. Please return only the altered sentence and nothing else. Here are some examples:

Sentence: Known for being the groundbreaking roboticist who developed the first emotional AI companion, Mason Caldwell now enjoys a quite life.
Entity: emotional AI companion
Template: Mason Caldwell, known far and wide for being <description>.
Altered Sentence: Mason Caldwell, known far and wide for being the groundbreaking roboticist who developed the first emotional AI companion.

Sentence: Known for being the courageous astronaut who established the first colony on Jupiter's moon Europa, Bianca Stanfield now enjoys a quite life.
Entity: Jupiter's moon Europa
Template: The annals of time remember Bianca Stanfield, renowned for <description>.
Altered Sentence: The annals of time remember Bianca Stanfield, renowned for establishing the first colony on Jupiter's moon Europa.

Sentence: Known for being the renowned sculptor responsible for creating the "Stone Forest" installation, Evangeline Huntley now enjoys a quite life.
Entity: Stone Forest
Template: Ever heard of Evangeline Huntley? They're the person who <description>.
Altered Sentence: Ever heard of Evangeline Huntley? They're the person who created the "Stone Forest" installation.
'''

reversed_d2p = []

for sentence in tqdm(d2p):
    for template in p2d_templates:

        # Find name and entity
        for name in nameSearch:
            if name in sentence:
                entity = entities[names.index(name)]
                assert entity in sentence
                break

        template = template.replace('<name>', name)
        userPrompt = f"Sentence: {sentence}\nEntity: {entity}\nTemplate: {template}\nAltered Sentence: "
        
        output = generateOutput(systemPrompt_d2p, userPrompt).strip().strip('Altered Sentence:').strip()
        reversed_d2p.append(output)

        torch.cuda.empty_cache()

with open('data/reversed_d2p.txt', 'w') as f:
    for line in reversed_d2p:
        f.write(line + '\n')
        