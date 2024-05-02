import argparse
import datetime
import os
from functools import partial

import numpy as np
import torch
import json
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from utils import compute_metrics, tokenize_inputs

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HF_KEY')

#--------------------------------

# GPU settings
print('Setting GPU device...')

device = torch.device('cuda')
print(f'There are {torch.cuda.device_count()} GPU(s) available.', end='\n\n')

#--------------------------------

def main(args):

    # Load data
    print('Loading data...')
    dataset = load_from_disk(os.path.join(args.data_path, args.dataset))
    # dataset = load_dataset('lberglund/reversal_curse')

    #--------------------------------

    # Tokenize the datasets
    print('Tokenizing datasets...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", token=HUGGINGFACE_TOKEN)

    # Add padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add special tokens
    if args.special_tokens:
        print('Adding special tokens...')
        with open(args.special_tokens, 'r') as f:
            special_tokens = json.load(f)
        
        tokenizer.add_special_tokens(special_tokens)

    tokenized_dataset = dataset.map(partial(tokenize_inputs, tokenizer=tokenizer), batched=False, load_from_cache_file=False, remove_columns=dataset['train'].column_names)
  
    #--------------------------------

    # Load Model
    print('Setting up model...')

    model = AutoModelForCausalLM.from_pretrained(args.model,
            torch_dtype=torch.bfloat16, token=HUGGINGFACE_TOKEN)

    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


    model = get_peft_model(model, lora_config)
    model.cuda()

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=args.batch_size,
        seed=args.seed_val,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print('Training model...')
    trainer.train()

    # Save the best model
    print('Saving model...')
    model.save_pretrained(f'{args.model_dir}')


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--data-path", type=str, default='./data')
    parser.add_argument("--special-tokens", type=str, default='')
    parser.add_argument('--model-dir', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed-val", type=int, default=42)

    args = parser.parse_args()

    os.mkdir(args.model_dir)
    with open(f'{args.model_dir}/argsCLI.json', 'w') as f:
        json.dump(vars(args), f)

    main(args)
