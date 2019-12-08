#!/usr/bin/env python3

"""
Script for evaluating a given file by a given model/NN.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from checkpoint_utils import load_checkpoint, checkpoint
from arg_parser import add_tensor_dataset_option


def parse_eval_args():
    parser = argparse.ArgumentParser()

    ## Data args
    parser.add_argument('--input-dir', dest='input_dir',
                        help='Input dataset directory (XML)',
                        default=None, type=str, metavar='DIR_PATH')

    add_tensor_dataset_option(parser)

    parser.add_argument('--output-dir', dest='output_dir',
                        help='Output directory',
                        default='.', type=str, metavar='DIR_PATH')

    ## Dataset args
    from arg_parser import add_dataset_args
    parser = add_dataset_args(parser)

    ## Generic model args
    from arg_parser import add_model_args
    add_model_args(parser)

    ## Other settings
    parser.add_argument('--model-path', dest='model_path',
                        help='Path to model checkpoint to use for evaluation',
                        action='append',
                        type=str, metavar='PATH')

    args = parser.parse_args()
    assert args.model_path is not None and len(args.model_path) > 0, \
            'At least one model must be provided'
    assert args.input_dir or (args.tensor_dataset and len(args.tensor_dataset) > 0), \
            'Either input-dir or tensor-dataset must be provided'

    return args


def evaluate_pytorch(model, dataloader, device=None) -> np.ndarray:

    model = model.eval()
    results = np.empty((0,))
    if device is not None:
        model.to(device)

    for *inputs, _ in tqdm(dataloader, leave=False):
        if device is not None:
            inputs = tuple(inp.to(device=device) for inp in inputs)

        # forward pass (prediction)
        with torch.no_grad():
            outputs = model(*inputs)
            results = np.append(results, outputs.cpu().numpy())

    return results


def evaluate_ensemble(models, dataloader, device=None) -> np.ndarray:
    preds = None
    for i, m in enumerate(models):
        results = evaluate_pytorch(m, dataloader, device=device)
        if preds is None:
            preds = np.zeros((len(models), len(results)))
        preds[i] = results

    return np.mean(preds, axis=0)


def write_hyperpartisan_predictions(predictions, eval_dataset, output_dir, cutoff=0.5):
    """Write predictions to output file following template from SemEval 2019 Task"""
    article_ids = [art.get_id() for art in eval_dataset.articles]

    ## Write to file
    with open(os.path.join(output_dir, 'prediction.txt'), 'w', encoding='utf-8') as out:
        for art_id, pred in zip(article_ids, predictions):
            print('{}\t{}\t{}'.format(
                art_id,
                'true' if pred > cutoff else 'false',
                ((pred - 0.5) * 2) if pred > 0.5 else ((0.5 - pred) * 2)),
                file=out
            )


def write_propaganda_predictions(predictions, eval_dataset, output_dir, cutoff=0.5):
    """Write predictions to output file following template from NLP4IF Task"""

    ## Write to file
    with open(os.path.join(output_dir, 'prediction.txt'), 'w', encoding='utf-8') as out:
        for idx, (art_idx, sent_idx) in enumerate(eval_dataset.samples):
            article_id = eval_dataset.articles[art_idx].get_id()

            print('{}\t{}\t{}'.format(
                article_id, sent_idx + 1,
                'propaganda' if predictions[idx] > cutoff else 'non-propaganda'),
                file=out
            )


def construct_eval_dataset(dataset_constructor, args):
    """
    Returns the constructed (eval) dataset and its shape.
    """

    if args.tensor_dataset:
        from serialize_pytorch_dataset import load_serialized_data

        Xs = list()
        Ys = list()
        for path in args.tensor_dataset:
            X, Y = load_serialized_data(path)
            Xs.append(X)
            Ys.append(Y)

        assert len({el.shape[0] for el in Xs + Ys}) == 1, 'All datasets must have the same batch dimension'
        return torch.utils.data.TensorDataset(*(Xs + [Ys[0]])), Xs[0].shape[1:]

    else:
        ## Load embeddings
        from arg_utils import load_embeddings
        embeddings = load_embeddings(args)

        dataset = dataset_constructor(args.input_dir, None, embeddings, args)
        return dataset, dataset.shape()


def construct_base_propaganda_dataset(articles_dir: str, truth_dir: str):
    from datasets.propaganda import PropagandaReader, PropagandaDataset
    reader = PropagandaReader(articles_dir, truth_dir)
    articles = reader.get_articles()
    return PropagandaDataset(articles)


def main():
    ## Parse command line args
    args = parse_eval_args()
    from pprint import PrettyPrinter
    PrettyPrinter(indent=4).pprint(vars(args))
    print()


    ## Use CUDA if available
    print('=> CUDA availability / use: "{}" / "{}"'.format(str(torch.cuda.is_available()), str(args.CUDA)))
    args.CUDA = args.CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.CUDA) else 'cpu')

    ## Dataset + Dataloader ## NOTE Update DATASET here
    from arg_utils import construct_hyperpartisan_flair_dataset, \
                        construct_propaganda_flair_dataset
    eval_dataset, input_shape = construct_eval_dataset(construct_propaganda_flair_dataset, args)

    from datasets import CachedDataset
    ## NOTE Use Cached dataset ?? (useful for ensemble runs, but not with TensorDataset)
    eval_dataloader = torch.utils.data.DataLoader(
        # CachedDataset(eval_dataset),
        eval_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.dataloader_workers,
        pin_memory=args.CUDA
    )

    ## Construct Model                                  ## NOTE Update MODEL here
    from nn_architectures import construct_cnn_bertha_von_suttner, \
                                 construct_hierarch_att_net, \
                                 construct_lstm, construct_AttnBiLSTM                                 
    model_constructor = construct_AttnBiLSTM

    ## Load models from checkpoints
    models = list()
    for m_path in args.model_path:
        model = model_constructor(input_shape[-1])
        load_checkpoint(m_path, model)
        models.append(model)

    ## Model Summary
    from torchsummary import summary
    print('\n ** Model Summary ** ')
    print(models[0], end='\n\n')

    ## Evaluate documents
    predictions = evaluate_ensemble(
        models, eval_dataloader, device=device
    )

    ## Write predictions to file
    # write_hyperpartisan_predictions(predictions, eval_dataset, args.output_dir)
    write_propaganda_predictions(
        predictions,
        eval_dataset if args.tensor_dataset is None else construct_base_propaganda_dataset(args.input_dir, None),
        ## PropagandaDataset must always be provided (even if a tensor-dataset is provided), to properly write predictions to output file
        args.output_dir
    )


if __name__ == '__main__':
    main()
