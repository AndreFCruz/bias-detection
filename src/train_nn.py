#!/usr/bin/env python3

"""
Train script for NN architectures.
Loads dataset lazily (as it's needed).
"""

import os
import time
import random
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from meter_utils import AverageMeter, ProgressMeter
from typing import Sequence
from checkpoint_utils import load_checkpoint, checkpoint
from nn_utils import binary_accuracy


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='Model\'s name')

    ## Data/Path args
    from arg_parser import add_train_data_args
    add_train_data_args(parser)

    ## Dataset args
    from arg_parser import add_dataset_args
    add_dataset_args(parser)

    ## Generic model args
    from arg_parser import add_model_args
    add_model_args(parser)

    ## Training args
    parser.add_argument('--epochs', dest='epochs',
                        help='Number of epochs to train for',
                        default=10, type=int, metavar='N')
    parser.add_argument('--resume', dest='resume',
                        help='Resume training from the given checkpoint (default: None)',
                        default=None, type=str, metavar='PATH')

    parser.add_argument('--hidden-dim', dest='hidden_dim',
                        help='Value for miscellaneous hidden dimensions',
                        default=None, type=int, metavar='N')

    ## Model args
    parser.add_argument('--test-percentage', dest='test_percentage',
                        help='Percentage of samples to use for testing',
                        default=0.4, type=float)

    parser.add_argument('--reduce-lr-patience', dest='reduce_lr_patience',
                        help='Number of non-improving epochs before reducing LR',
                        default=5, type=int)

    parser.add_argument('--freeze', dest='freeze',
                        help='Indices of layers/sub-modules to freeze',
                        action='append', type=int)

    parser.add_argument('--k-fold', dest='k_fold',
                        help='Number of fold to use if using k-fold cross-validation',
                        default=None, type=int, metavar='K')

    ## Other settings
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir',
                        help='Directory for saving model checkpoints',
                        default='checkpoints', type=str)
    parser.add_argument('--print-freq', dest='print_freq',
                        help='Frequency of epochs for printing train statistics',
                        default=1, type=int, metavar='N')
    parser.add_argument('--validate-every', dest='validate_every',
                        help='Frequency of evaluation on validation dataset (in epochs)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--checkpoint-every', dest='checkpoint_every', ## Deprecated argument
                        help='Frequency of checkpointing (in epochs)',
                        default=None, type=int, metavar='N')


    ## Parse Args and Check Constraints
    args = parser.parse_args()
    if not args.tensor_dataset: ## If no tensor_dataset was provided
        assert args.granularity is not None and len(args.granularity) > 0, \
            'Please select a level of granularity for embeddings representations'
        assert args.embeddings_path is None or (args.embeddings_matrix_path is None and args.word2index_path is None), \
            'Invalid embeddings settings'

    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('\n')

    return args


def train_pytorch(model, optimizer, loss_criterion, train_loader, args,
                  val_loader=None, device=None, scheduler=None):

    ## Statistics
    batch_time_meter    = AverageMeter('BatchIterTime', ':6.3f')
    data_time_meter     = AverageMeter('BatchLoadTime', ':6.3f')
    loss_meter          = AverageMeter('TrainLoss', ':.4e')
    acc_meter           = AverageMeter('TrainAcc', ':6.2f')
    meters = [
        batch_time_meter, data_time_meter, loss_meter, acc_meter
    ]
    progress = ProgressMeter(args.epochs, *meters)

    model = model.train()
    if device is not None:
        model.to(device)
    print('Batch size: {:3} ; Train batches: {:3} ; Val batches: {:3}'.format(
                args.batch_size, len(train_loader),
                len(val_loader) if val_loader is not None else '---'))

    best_val_acc, best_model_path = None, None
    for epoch in range(1, args.epochs + 1):
        for m in meters:
            m.reset()

        running_loss = 0
        start = time.time()
        for *inputs, labels in tqdm(train_loader,
                                    desc='Epoch {:2}'.format(epoch),
                                    ncols=100,
                                    leave=False):
            data_time_meter.update(time.time() - start)

            # move batch to device in use
            if device is not None:
                labels = labels.to(device=device)
                inputs = tuple(inp.to(device=device) for inp in inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(*inputs)
            # labels = labels.type(outputs.type())

            # propagate backwards + step optimizer
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate and store statistics
            with torch.no_grad():
                train_acc = accuracy_score(labels.cpu(), outputs.round().cpu())

            all_batch_sizes = {el.size(0) for el in inputs}
            batch_size = next(iter(all_batch_sizes))
            assert len(all_batch_sizes), 'All inputs should have the same batch_size'

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(train_acc, batch_size)
            running_loss += loss.item()

            # measured elapsed time
            batch_time_meter.update(time.time() - start)

            # reset timer for next batch
            start = time.time()

        # Per epoch:
        epoch_loss = running_loss / len(train_loader)
        if epoch % args.print_freq == 0:
            progress.print(epoch,
                end='\t FinalTrainLoss {:.4e}\n'.format(epoch_loss))

        if args.validate_every is not None and val_loader is not None \
            and epoch % args.validate_every == 0:
            val_acc, val_loss = binary_accuracy(model, val_loader, loss_criterion=loss_criterion, device=device)
            print('\t=> Val Acc: {:.2f} ; Val Loss: {:.4e}'.format(val_acc, val_loss))

            # If this is the best val loss so far, checkpoint model
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                model_name = '{}.val-acc-{:.3f}'.format(args.name, val_acc)
                best_model_path = checkpoint(model,
                                             optimizer=optimizer, scheduler=scheduler,
                                             epoch=epoch, checkpoint_dir=args.checkpoint_dir,
                                             name=model_name)

            model = model.train()

        if args.checkpoint_every is not None and epoch % args.checkpoint_every == 0:
            checkpoint(model, optimizer=optimizer, scheduler=scheduler,
                       epoch=epoch, checkpoint_dir=args.checkpoint_dir,
                       name=args.name)

        if scheduler is not None:
            scheduler.step(epoch_loss)

    ## Return best_model and respective metric
    return best_model_path, best_val_acc


def freeze_layers(model, indices_to_freeze: Sequence[int]):
    for idx, child in enumerate(model.children()):
        if idx in indices_to_freeze:
            print('Freezing following layer/sub-module ({}):'.format(idx))
            print(child)

            for param in child.parameters():
                param.requires_grad = False
            child.eval()


def construct_datasets(dataset_constructor, embeddings, args):
    ## Train dataset
    if args.tensor_dataset:
        from serialize_pytorch_dataset import load_serialized_data
        input_shape = None

        Xs = list()
        Ys = list()
        for path in args.tensor_dataset:
            X, Y = load_serialized_data(path)
            Xs.append(X)
            Ys.append(Y)
            if input_shape is None:
                input_shape = X.shape[1:]

        assert len({el.shape[0] for el in Xs + Ys}) == 1, 'All datasets must have the same batch dimension'
        train_dataset = TensorDataset(*(Xs + [Ys[0]]))

    elif args.train_news_dir is not None and args.train_ground_truth_dir is not None:
        from datasets import CachedDataset
        train_dataset = dataset_constructor(args.train_news_dir, args.train_ground_truth_dir, embeddings, args)
        input_shape = train_dataset.shape()
        train_dataset = CachedDataset(train_dataset)

    else:
        train_dataset = None
        print('\n=> # NOT USING A TRAIN DATASET; ONLY FOR TESTING #\n')

    ## Val dataset
    if args.val_news_dir is None or args.val_ground_truth_dir is None:
        val_dataset = None
    else:
        from datasets import CachedDataset
        val_dataset = dataset_constructor(args.val_news_dir, args.val_ground_truth_dir, embeddings, args)
        val_dataset = CachedDataset(val_dataset)

    ## Test dataset: if no test dataset was provided, split train dataset
    if args.test_news_dir is None or args.test_ground_truth_dir is None and train_dataset is not None:
        print('=> No test dataset provided, splitting train/test {}/{}'.format(
                1. - args.test_percentage, args.test_percentage))

        num_train = int(len(train_dataset) * (1 - args.test_percentage) + 0.5)
        num_test  = int(len(train_dataset) * args.test_percentage + 0.5)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_test])
    else:
        print('Test dataset: "{}"'.format(args.test_news_dir))
        test_dataset = dataset_constructor(args.test_news_dir, args.test_ground_truth_dir, embeddings, args)
        input_shape = test_dataset.shape()

    return train_dataset, val_dataset, test_dataset, input_shape


def construct_dataloaders(train_dataset, val_dataset, test_dataset, args):
    ## Train dataloader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.dataloader_workers,
                    ## 0 -> main_thread; >0 -> number of separate threads
                    ## warning: debugging gets bugged when using separate threads :)
        pin_memory=args.CUDA
    ) if train_dataset is not None else None

    ## Validation dataloader
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.CUDA
    ) if val_dataset is not None else None

    ## Test dataloader
    assert test_dataset is not None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.CUDA)

    return train_dataloader, val_dataloader, test_dataloader


def construct_model(model_constructor, input_shape, args):
    ## Construct model
    model = model_constructor(input_shape[-1], args)
    if torch.cuda.is_available() and args.CUDA:
        print('=> Moving model to CUDA')
        model.cuda()

    ## Construct optimizer
    from optimizers import RAdam, AdaBound
    # optimizer = RAdam(model.parameters())   ## NOTE Experimenting with RAdam and AdaBound
    # optimizer = AdaBound(model.parameters())   ## NOTE Experimenting with RAdam and AdaBound
    optimizer = torch.optim.Adam(model.parameters())

    ## Construct scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2,
        patience=args.reduce_lr_patience,
        verbose=True
    )

    ## Optionally, resume training from checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        print('\n=> ** Resuming training from checkpoint **')
        load_checkpoint(args.resume, model, optimizer, scheduler)

    ## Optionally, freeze specific layers/sub-modules
    if args.freeze is not None and len(args.freeze) > 0:
        print('=> Freezing layers/sub-modules with indices: {}'.format(args.freeze))
        freeze_layers(model, args.freeze)
    else:
        print('=> All layers unfrozen for training')

    ## Model summary #1
    from nn_utils import count_parameters
    print('Model has {} trainable parameters'.format(count_parameters(model)))
    print(model)

    ## Model Summary #2
    # from torchsummary import summary
    # print('\nModel Summary:')
    # summary(model, input_shape, device='cuda' if args.CUDA else 'cpu')

    ## Loss criterion: Binary Cross Entropy
    loss_criterion = torch.nn.BCELoss()

    return model, optimizer, scheduler, loss_criterion


def test_model(model, test_dataloader, device=None):
    from nn_utils import evaluate_metric
    from sklearn.metrics import classification_report, accuracy_score, \
                                precision_score, recall_score, f1_score

    device = device if device is not None else torch.device('cpu')
    report = evaluate_metric(model, test_dataloader, classification_report, device=device)
    print('\n** Classification Report **')
    print(report)

    acc = evaluate_metric(model, test_dataloader, accuracy_score, device=device)
    prec = evaluate_metric(model, test_dataloader, precision_score, device=device)
    rec = evaluate_metric(model, test_dataloader, recall_score, device=device)
    f1 = evaluate_metric(model, test_dataloader, f1_score, device=device)
    print('A: {:.2%} ; P: {:.2%} ; R: {:.2%} ; F1: {:.2%}'.format(acc, prec, rec, f1))
    return acc, prec, rec, f1


def main():
    ## Parse command line args
    args = parse_train_args()

    ## Use CUDA if available
    print('=> CUDA availability / use: "{}" / "{}"'.format(str(torch.cuda.is_available()), str(args.CUDA)))
    args.CUDA = args.CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.CUDA) else 'cpu')

    ## Load embeddings
    from arg_utils import load_embeddings
    embeddings = load_embeddings(args)

    ## Construct Datasets (Train/Validation/Test)
    from arg_utils import construct_hyperpartisan_flair_dataset, \
                          construct_hyperpartisan_flair_and_features_dataset, \
                          construct_propaganda_flair_dataset
    train_dataset, val_dataset, test_dataset, input_shape = construct_datasets(
        construct_propaganda_flair_dataset,
        embeddings, args,
    )

    ## Construct Dataloaders
    train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(
        train_dataset, val_dataset, test_dataset, args
    )

    ## Construct model + optimizer + scheduler
    from nn_architectures import construct_hierarch_att_net, \
                                 construct_cnn_bertha_von_suttner, \
                                 construct_HAN_with_features, \
                                 construct_lstm
    model, optimizer, scheduler, loss_criterion = construct_model(
        construct_lstm,     ## NOTE change model here
        input_shape, args
    )

    ## Train model
    if train_dataloader is not None:
        random.seed(args.seed)
        best_path, _ = train_pytorch(
            model, optimizer, loss_criterion, train_dataloader, args=args,
            val_loader=test_dataloader if val_dataloader is None else val_dataloader,
            device=device, scheduler=scheduler
        )
        checkpoint(model, optimizer, scheduler, args.epochs, args.checkpoint_dir, name='final.' + args.name)

    ## Test model
    # Load best model's checkpoint for testing (if available)
    if best_path:
        load_checkpoint(best_path, model)
    test_model(model, test_dataloader, device=device)


if __name__ == '__main__':
    main()
