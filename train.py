import os
import gc
import json
import argparse
import datetime
import numpy as np
import seaborn as sns
from copy import deepcopy
from dotenv import load_dotenv
import matplotlib.pyplot as plt

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

#--------------------------------

# GPU settings
print('Setting GPU device...')

device = torch.device('cuda')
print(f'There are {torch.cuda.device_count()} GPU(s) available.', end='\n\n')

#--------------------------------

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument('--model-dir', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed-val", type=int, default=42)

    args = parser.parse_args()

    os.mkdir(args.model_dir)
    with open(f'{args.model_dir}/argsCLI.json', 'w') as f:
        json.dump(vars(args), f)

    #--------------------------------

    # Load data
    print('Loading data...')
    # dataset = load_from_disk(os.path.join(args.data_path, args.dataset)
    dataset = load_dataset('lberglund/reversal_curse')

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', token=os.getenv('HF_KEY'))

    # Add padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def tokenize_function(examples):
        prompts = tokenizer(examples['prompt'], padding='max_length', truncation=True)
        completions = tokenizer(examples['completion'], padding='max_length', truncation=True)
        return {
            'input_ids': prompts['input_ids'],
            'attention_mask': prompts['attention_mask'],
            'labels': completions['input_ids']
        }

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    #--------------------------------

    # Load Model
    print('Setting up model...')

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.cuda()

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed_val,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=
    )

    trainer.train()

    # Save the best model
    print('Saving model...')
    model.save_pretrained(f'{args.model_dir}')
