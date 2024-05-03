import argparse
import datetime
import os
from functools import partial
from transformers import TrainerCallback

import numpy as np
import torch
import json
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HF_KEY')


def tokenize_inputs_train(dataset, tokenizer):
    tokenized_inputs = tokenizer(dataset['prompt'] + dataset['completion'])
    return tokenized_inputs


def tokenize_inputs_val(dataset, tokenizer):

    tokenized_prompt = tokenizer(dataset['prompt'])
    tokenized_completion = tokenizer(dataset['completion'])

    tokenized_data = {}

    tokenized_data['eval_labels'] = torch.tensor([-100 for _ in range(len(tokenized_prompt['input_ids']))] + tokenized_completion['input_ids'])
    tokenized_data['input_ids'] = torch.tensor(tokenized_prompt['input_ids'] + tokenized_completion['input_ids'])
    tokenized_data['attention_mask'] = torch.tensor(tokenized_prompt['attention_mask'] + tokenized_completion['attention_mask'])

    return tokenized_data


def compute_metrics(eval_pred):

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side="right", token=HUGGINGFACE_TOKEN)
    logits, labels = eval_pred

    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]


    labels_to_predict = shifted_labels != -100

    # print(shifted_labels[0].shape)
    # print(tokenizer.decode(shifted_labels[0]*labels_to_predict[0]))
    # print(tokenizer.decode(shifted_labels[1]*labels_to_predict[1]))

    num_datapoints = np.sum(labels_to_predict)
    predictions = np.argmax(shifted_logits, axis=-1)

    num_correctly_predicted = np.sum((predictions == shifted_labels) * labels_to_predict)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fn(torch.from_numpy(shifted_logits.reshape(-1, shifted_logits.shape[-1])),
                torch.from_numpy(shifted_labels.reshape(-1)))

    return {
        'accuracy': num_correctly_predicted/num_datapoints,
        'loss': loss
    }

class CustomTrainerCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, device):
        
        super().__init__()
        self.eval_dataset = eval_dataset
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        # Access the eval_dataloader and perform actions
        # For example, evaluate the model on the eval_dataloader
        self.evaluate(kwargs['model'])
        # Do something with the eval_result, like logging or saving

    
    def evaluate(self, model):

        accuracy = 0
        total_loss = 0
        num_datapoints = 0

        for data in self.eval_dataset:

            with torch.no_grad():
                logits = model.forward(data['input_ids'].to(self.device).unsqueeze(0), data['attention_mask'].to(self.device).unsqueeze(0)).logits.to('cpu')
            labels = data['eval_labels'].unsqueeze(0)

            shifted_logits = logits[..., :-1, :]
            shifted_labels = labels[..., 1:]

            labels_to_predict = shifted_labels != -100

            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side="right", token=HUGGINGFACE_TOKEN)

            predictions = np.argmax(shifted_logits, axis=-1)


            num_datapoints = torch.sum(labels_to_predict)
            num_correctly_predicted = torch.sum((predictions == shifted_labels) * labels_to_predict)

            # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            # loss = loss_fn(torch.from_numpy(shifted_logits.reshape(-1, shifted_logits.shape[-1])),
            #             torch.from_numpy(shifted_labels.reshape(-1)))
            print('heeeee')
            print(num_datapoints)
            print(num_correctly_predicted)
            #print(tokenizer.decode(predictions[0]))
            input_text = "Who is the director of \"A Journey Through Time\"?"
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(input_ids.to(self.device), max_new_tokens=10)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(output_text)


        return

        for data in self.eval_dataset:

            logits = model(data['input_ids'], data['attention_mask'])
            labels = torch.tensor(data['labels'])

            print(data, logits)

        return

        print(eval_pred)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side="right", token=HUGGINGFACE_TOKEN)
        logits, labels = sel

        shifted_logits = logits[..., :-1, :]
        shifted_labels = labels[..., 1:]


        labels_to_predict = shifted_labels != -100

        print(shifted_labels[0].shape)
        print(tokenizer.decode(shifted_labels[0]*labels_to_predict[0]))
        print(tokenizer.decode(shifted_labels[1]*labels_to_predict[1]))

        num_datapoints = np.sum(labels_to_predict)
        predictions = np.argmax(shifted_logits, axis=-1)

        num_correctly_predicted = np.sum((predictions == shifted_labels) * labels_to_predict)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fn(torch.from_numpy(shifted_logits.reshape(-1, shifted_logits.shape[-1])),
                    torch.from_numpy(shifted_labels.reshape(-1)))

        return {
            'accuracy': num_correctly_predicted/num_datapoints,
            'loss': loss
        }