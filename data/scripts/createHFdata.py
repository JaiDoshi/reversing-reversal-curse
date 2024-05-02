import json
from tqdm import tqdm
from datasets import load_dataset

experimentsPath = './data/nlu_experiments'
trainingPath = './data/nlu_experiments/training'
valTestPath = './data/nlu_experiments/validation-test'

structure = {
    'Exp1_A_Original': {
        'train': ['both_prompts_train.jsonl', 'd2p_prompts_train.jsonl', 'p2d_prompts_train.jsonl'],
        'validation': ['validation_prompts.jsonl'],
        'test': ['d2p_prompts_test.jsonl', 'p2d_prompts_test.jsonl']
    },
    'Exp1_B_Meta_Augment': {
        'train': ['both_tokens_train.jsonl', 'd2p_prompts_train.jsonl', 'p2d_prompts_train.jsonl'],
        'validation': ['validation_prompts.jsonl'],
        'test': ['d2p_prompts_test.jsonl', 'p2d_prompts_test.jsonl']
    }
    # 'Exp1_C_Full_Augment': {
    #     'train': ['both_prompts_train.jsonl'],
    #     'validation': ['validation_prompts.jsonl],
    #     'test': ['d2p_prompts_test.jsonl', 'p2d_prompts_test.jsonl']
    # },
    # 'Exp1_D_HalfnHalf': {
    #     'train': ['both_prompts_train.jsonl'],
    #     'validation': ['validation_prompts.jsonl],
    #     'test': ['d2p_prompts_test.jsonl', 'p2d_prompts_test.jsonl']
    # }
}

#---------------------

datadicts = {}
for expName, expStructure in tqdm(structure.items()):
    datadicts[expName] = {}

    # Load data
    dataset = load_dataset(
        'json',
        data_files={
            'train': [f'{trainingPath}/{file}' for file in expStructure['train']],
            'validation': [f'{valTestPath}/{file}' for file in expStructure['validation']],
            'test': [f'{valTestPath}/{file}' for file in expStructure['test']]
        },
    )

    # Save data as datadict
    dataset.save_to_disk(f'{experimentsPath}/{expName}/dataset')