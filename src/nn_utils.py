"""
Several util functions for use in different neural-network related modules.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Union, Callable, Sequence, Any


def balanced_sampling(y):
    """
    Returns indices of samples to use in a balanced dataset.
    """
    y = np.round(y, decimals=0).astype(int)
    if len(y.shape) > 1:
        y = y.flatten()
    count = np.bincount(y)
    class_indices = {}
    for label in range(len(count)):
        class_indices[label] = np.argwhere(y == label).flatten()

    minority_class = count.min()
    balanced_indices = np.empty((minority_class * len(count)), dtype=int)
    count = 0
    for key in class_indices:
        for idx in np.random.choice(class_indices[key], size=minority_class, replace=False):
            balanced_indices[count] = idx
            count += 1
    
    np.random.shuffle(balanced_indices)
    return balanced_indices


def generate_random_unit_vec(shape: Tuple[int]):
    vec = np.random.rand(*shape)
    return vec / np.linalg.norm(vec)


def flatten(t: Union[np.ndarray, torch.Tensor]):
    """Flatten a Tensor/ndarray object"""
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def count_parameters(model):
    """
    Counts the number of trainable parameters on the given model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_metric(model: torch.nn.Module,
                    dataloader: DataLoader,
                    metric: Callable[[Sequence, Sequence], Any],
                    device=None):
    """
    Evaluates the given model on the given metric.
    Compatible with most metrics from sklearn.metrics
    """
    model.eval()

    y_true, y_pred = list(), list()
    for *inputs, labels in dataloader:
        # move batch to device in use
        if device is not None:
            labels = labels.to(device=device)
            inputs = tuple(inp.to(device=device) for inp in inputs)

        with torch.no_grad():
            outputs = model(*inputs).round()

        y_pred.extend(map(int, outputs.cpu().numpy()))
        y_true.extend(map(int, labels.cpu().numpy()))
    
    return metric(y_true, y_pred)


def accuracy_topk(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    Useful for multiclass classification (e.g. imagenet)
    From: https://github.com/pytorch/examples/blob/5df464c46cf321ed1cc3df1e670358d7f5ae1887/imagenet/main.py#L399
    """
    assert len(output.shape) > 1 ## This is for multiclass classification, from model outputs of distribution over classes

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def binary_accuracy(model: torch.nn.Module, dataloader, loss_criterion=None, device=None):
    """
    Calculate (binary) accuracy on a given dataloader.
    If provided with a loss criterion, also returns the loss on the dataset.
    """
    model.eval()

    num_correct     = 0
    num_examples    = 0
    running_loss    = 0.0
    for *inputs, labels in dataloader:
        # move batch to device in use
        if device is not None:
            labels = labels.to(device=device)
            inputs = tuple(inp.to(device=device) for inp in inputs)

        with torch.no_grad():
            outputs = model(*inputs)
        if loss_criterion is not None:
            running_loss += loss_criterion(outputs, labels).item()

        correct = torch.eq(torch.round(outputs).type(labels.type()), labels).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]

    if loss_criterion is None:
        return num_correct / num_examples
    else:
        return num_correct / num_examples, running_loss / len(dataloader)
