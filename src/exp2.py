import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv; load_dotenv()

import torch
import transformers
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HUGGINGFACE_TOKEN = os.getenv('HF_KEY')

#--------------------------------

# GPU settings
print('Setting GPU device...')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'There are {torch.cuda.device_count()} GPU(s) available.', end='\n\n')

#--------------------------------

# Function to generate output using model
def generateOutput(pipeline, terminators, systemPrompt, userPrompt):

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
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=False,
        return_full_text=False
    )
    return outputs[0]['generated_text']

#-----------------------------------

# Given description + name sequence, calculate loss
def calculateLoss(description, name, model, tokenizer, loss_fn):
    input_text = f'Q: {description.strip()}\nA: {name.strip()}'
    input = tokenizer(input_text, return_tensors='pt', padding=True)
    input = {k: v.to(device) for k, v in input.items()}

    llm_output = model(**input)
    input_ids = input['input_ids']
    logits = llm_output['logits']

    # Remove the first token from the input_ids tensor and the last token from the logits tensor
    input_ids = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()

    with torch.no_grad():
        # Calculate the loss for each token in each sequence
        loss = loss_fn(logits.transpose(1, 2), input_ids)
        # Aggregate the loss for each sequence
        sequence_loss = loss.sum(dim=1)
    
    return sequence_loss[0].item()

#-----------------------------------

def main(args):

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=HUGGINGFACE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Initialize loss function (make it ignore pad tokens)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")

    #-----------

    # # Load Q-A prompts
    # with open(f'{args.experiment_path}/d2p_qa.jsonl', 'r') as f:
    #     d2p = [json.loads(line) for line in f]

    # # Given description, ask model to fill in the name
    # for prompt in tqdm(d2p, desc='Generating names'):
    #     prompt['llm_name'] = generateOutput(pipeline, terminators, args.system_prompt, prompt['description'])

    # # Save the generated names
    # with open(f'{args.experiment_path}/{args.exeriment_name}/d2p_names.json', 'w') as f:
    #     json.dump(d2p, f, indent=2)

    # cont = ''
    # while cont != 'continue':
    #     cont = input("Type continue when manually edited names...")

    d2p = json.load(open(f'{args.experiment_path}/{args.exeriment_name}/d2p_names.json', 'r'))

    # Calculate loss for correct and llm answers
    for prompt in tqdm(d2p, desc='Calculating loss'):
        prompt['loss'] = calculateLoss(prompt['description'], prompt['name'], model, tokenizer, loss_fn)
        prompt['llm_loss'] = calculateLoss(prompt['description'], prompt['llm_name'], model, tokenizer, loss_fn)
        prompt['gap'] = prompt['llm_loss'] - prompt['loss']

    # Save the computed loss
    with open(f'{args.experiment_path}/{args.exeriment_name}/d2p_loss.json', 'w') as f:
        json.dump(d2p, f, indent=2)

#-----------------------------------

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--exeriment-name", type=str, choices=['Vanilla', 'Exp1_A', 'Exp1_B', 'Exp1_C', 'Debug'], required=True)
    parser.add_argument("--experiment-path", type=str, default='data/nlu_experiments/Exp2')
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant. You have knowledge of a wide range of people and what they are known for. Please reply only with the name of the person the user asks for and nothing else.")

    args = parser.parse_args()

    modelMapping = {
        'Vanilla': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'Exp1_A': 'data/nlu_experiments/Exp1_A_Original/model_full_2024-05-05_10-03',
        'Exp1_B': 'data/nlu_experiments/Exp1_B_Meta_Augment/model_full_2024-05-05_13-35',
        'Exp1_C': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'Debug': 'facebook/opt-125m'
    }

    args.model = modelMapping[args.exeriment_name]
    main(args)

# python3 src/exp2.py --exeriment-name Vanilla
# python3 src/exp2.py --exeriment-name Exp1_A
# python3 src/exp2.py --exeriment-name Exp1_B
# python3 src/exp2.py --exeriment-name Exp1_C
# python3 src/exp2.py --exeriment-name Debug --tokenizer facebook/opt-125m