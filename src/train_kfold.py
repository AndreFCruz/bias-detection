#!/usr/bin/env python3

"""
Script for training a pytorch module with k-fold cross-validation.
Uses functions from train_nn.py
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from nn_utils import balanced_sampling
from train_nn import                \
    parse_train_args,               \
    train_pytorch,                  \
    test_model,                     \
    construct_model
from checkpoint_utils import load_checkpoint


def extract_data(args):
    if args.tensor_dataset:
        from serialize_pytorch_dataset import load_serialized_data

        Xs = list()
        Ys = list()
        for path in args.tensor_dataset:
            X, Y = load_serialized_data(path)
            Xs.append(X)
            Ys.append(Y)

        return tuple(Xs) + (Ys[0],)
    else:
        ## Load embeddings
        from arg_utils import load_embeddings
        embeddings = load_embeddings(args)

        ## Main dataset (for train/test)
        from arg_utils import construct_hyperpartisan_flair_dataset, \
                              construct_hyperpartisan_flair_and_features_dataset
        dataset_constructor = construct_hyperpartisan_flair_dataset   ## NOTE Update DATASET here
        main_dataset = dataset_constructor(args.train_news_dir, args.train_ground_truth_dir, embeddings, args)
        
        return extract_whole_dataset(main_dataset, args.dataloader_workers)


def extract_whole_dataset(dataset, dataloader_workers=0):
    """
    Converts the given pytorch Dataset into a Tensor.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available()
    )

    X, Y = next(iter(dataloader))
    return X, Y


def main():
    ## Parse command line args
    args = parse_train_args()
    assert args.k_fold is not None, 'Use "--k-fold <N>" for specifying the number of folds to use'

    ## Use CUDA if available
    print('=> CUDA availability / use: "{}" / "{}"'.format(str(torch.cuda.is_available()), str(args.CUDA)))
    args.CUDA = args.CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.CUDA) else 'cpu')

    ## Extract data
    *Xs, Y = extract_data(args)
    Xs = tuple(Xs)
    input_shape = Xs[0].shape[1:]

    ## Optionally, undersample majority class
    if args.undersampling:
        balanced_indices = balanced_sampling(Y)
        Y = Y[balanced_indices]
        Xs = tuple(x[balanced_indices] for x in Xs)

    ## Construct TensorDataset
    main_tensor_dataset = TensorDataset( *(Xs + (Y,)) )

    ## Model constructor ## NOTE Change MODEL here
    from nn_architectures import construct_lstm, construct_AttnBiLSTM
    model_constructor = construct_AttnBiLSTM

    ## k-fold split of Train/Test
    stats = np.zeros((args.k_fold, 4))
    kfold = StratifiedKFold(n_splits=args.k_fold)
    for i, (train_indices, test_indices) in enumerate(kfold.split(Xs[0].numpy(), Y.numpy())):
        print('K-Fold: [{:02}/{:02}]'.format(i + 1, args.k_fold))

        train_dataset, test_dataset = Subset(main_tensor_dataset, train_indices), Subset(main_tensor_dataset, test_indices)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available())
        
        ## Construct model + optimizer + scheduler
        args.name = args.name + '.k{}'.format(i)
        model, optimizer, scheduler, loss_criterion = construct_model(
            model_constructor, input_shape, args
        )

        ## Train model
        best_model, _ = train_pytorch(
            model, optimizer, loss_criterion, train_loader, args=args,
            val_loader=test_loader, device=device, scheduler=scheduler
        )

        ## Load best model
        if best_model:
            load_checkpoint(best_model, model)

        stats[i] = test_model(model, test_loader, device)


    ## Final stats
    print('** Final Statistics **')
    print('Mean:\t', np.mean(stats, axis=0))
    print('STD: \t', np.std(stats, axis=0))


if __name__ == '__main__':
    main()
