import os
import json
import argparse
import datetime
from functools import partial
from dotenv import load_dotenv; load_dotenv()

import torch
from datasets import load_from_disk
from utils import compute_metrics, tokenize_inputs
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

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
    dataset = load_from_disk(f'{args.experiment_path}/dataset')

    #--------------------------------

    # Tokenize the datasets
    print('Tokenizing datasets...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", token=HUGGINGFACE_TOKEN)

    # Add padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add special tokens
    print('Adding special tokens...')
    with open(args.special_tokens, 'r') as f:
        special_tokens = json.load(f)

    special_tokens = {'additional_special_tokens': special_token for special_token in special_tokens.values()}
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
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        seed=args.seed_val,
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
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
    parser.add_argument("--experiment-path", type=str, required=True)
    parser.add_argument("--special-tokens", type=str, default='data/raw/tokenMapping.json')
    parser.add_argument('--model-dir', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed-val", type=int, default=42)

    args = parser.parse_args()

    # Create model directory
    args.model_dir = f'{args.experiment_path}/{args.model_dir}'
    os.mkdir(args.model_dir)
    
    with open(f'{args.model_dir}/argsCLI.json', 'w') as f:
        json.dump(vars(args), f)

    main(args)
