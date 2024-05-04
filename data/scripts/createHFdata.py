import json
from tqdm import tqdm
from datasets import load_dataset

experimentsPath = './data/nlu_experiments'
trainingPath = './data/nlu_experiments/training'
valTestPath = './data/nlu_experiments/validation-test'

structure = {
    'Exp1_A_Original': {
        'meta-dataset': {
            'train': ['both_prompts_train.jsonl']
        },
        'dataset': {
            'train': ['d2p_prompts_train.jsonl', 'p2d_prompts_train.jsonl']
        }
    },
    'Exp1_B_Meta_Augment': {
        'meta-dataset': {
            'train': ['both_tokens_train.jsonl']
        },
        'dataset': {
            'train': ['d2p_prompts_train.jsonl', 'p2d_prompts_train.jsonl']
        }
    },
    # 'Exp1_C_Full_Augment': {
    #     'meta-dataset': {
    #         'train': ['both_prompts_train.jsonl']
    #     },
    #     'dataset': {
    #         'train': ['d2p_tokens_train.jsonl', 'p2d_tokens_train.jsonl']
    #     }
    # },
}

#---------------------

# Create datasets
for expName, expStructure in tqdm(structure.items()):
    for datasetType, datasetFiles in expStructure.items():

        # Load data
        dataset = load_dataset(
            'json',
            data_files={
                'train': [f'{trainingPath}/{file}' for file in datasetFiles['train']]
            },
        )

        # Save data as datadict
        dataset.save_to_disk(f'{experimentsPath}/{expName}/{datasetType}')
