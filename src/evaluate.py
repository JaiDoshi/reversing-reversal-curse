import os
import json
import argparse
import datetime
from functools import partial
from dotenv import load_dotenv; load_dotenv()
import pandas as pd

import torch
from datasets import load_from_disk
from utils import compute_metrics, tokenize_inputs
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, pipeline)
import sys

HUGGINGFACE_TOKEN = os.getenv('HF_KEY')

#--------------------------------

# GPU settings
print('Setting GPU device...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#--------------------------------

def generate_output(prompt, pipe, generate_kwargs):
    return pipe(prompt, **generate_kwargs)

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right", token=HUGGINGFACE_TOKEN)

    # Add padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print('Setting up model...')
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, token=HUGGINGFACE_TOKEN)
    model.to(device)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    generate_kwargs = {'max_new_tokens': 30}

    files = {
        'd2p_qa_test': 'data/nlu_experiments/validation-test/d2p_qa_test.json',
        'p2d_qa_test': 'data/nlu_experiments/validation-test/p2d_qa_test.json'
    }

    for fileType, file in files.items():

        data = json.load(open(file))
        for entry in data:
            output = generate_output(entry['p2d']['question'], pipe, generate_kwargs)
            entry['p2d']["predicted_answer"] = output[0]['generated_text']

            output = generate_output(entry['d2p']['question'], pipe, generate_kwargs)
            entry['d2p']["predicted_answer"] = output[0]['generated_text']

        df = pd.DataFrame([
            {
                **{'name': item['name'], 'entity': item['entity'], 'prompt_example': item['prompt_example'], 'direction': key},
                **value
            }
            for item in data
            for key, value in item.items() if key in ['p2d', 'd2p']
        ])

        df.to_csv(os.path.join(args.experiment_path, f'{fileType}_outputs.csv'), index=False)

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--experiment-path', type=str, required=True)
    args = parser.parse_args()
    main(args)

#python src/evaluate.py --model-dir data/nlu_experiments/Exp1_A_Original/model_2024-05-03_12-33 --experiment-path data/nlu_experiments/Exp1_A_Original
#python src/evaluate.py --model-dir data/nlu_experiments/Exp1_B_Meta_Augment/model_2024-05-04_09-00 --experiment-path data/nlu_experiments/Exp1_B_Meta_Augment