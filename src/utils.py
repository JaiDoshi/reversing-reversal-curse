import numpy as np
import torch


def tokenize_inputs(dataset, tokenizer):
    return tokenizer(dataset['prompt'] + dataset['completion'])


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

    labels_to_predict = shifted_labels != -100
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