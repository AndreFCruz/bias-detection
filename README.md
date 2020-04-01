# Bias Detection in Texts

State of the art _propaganda_ and _hyperpartisan_ news detection with PyTorch.

## Some cool plots
> Attention plots: blue indicates sentence-wise attention, and red token-wise.

| Example Hyperpartisan Article | Example Mainstream Article |
|:-:|:-:|
| ![Hyperpartisan Article](./imgs/176-single-col.png) | ![Mainstream Article](./imgs/481-single-col.png) |


## Getting started
On a new **Python3.7** virtualenv, run:
```
pip install -r requirements.txt
```

## Citing
Code used in the following papers:
* [On Document Representations for Detection of Biased News Articles](https://dl.acm.org/doi/abs/10.1145/3341105.3374025)
* [On Sentence Representations for Propaganda Detection: From Handcrafted Features to Word Embeddings](https://www.aclweb.org/anthology/D19-5015/)
* [Team Fernando-Pessa at SemEval-2019 Task 4: Back to Basics in Hyperpartisan News Detection](https://www.aclweb.org/anthology/S19-2173/)

Bibtex (for latest paper):
```
@inproceedings{10.1145/3341105.3374025,
  author = {Cruz, Andr\'{e} Ferreira and Rocha, Gil and Cardoso, Henrique Lopes},
  title = {On Document Representations for Detection of Biased News Articles},
  year = {2020},
  isbn = {9781450368667},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3341105.3374025},
  doi = {10.1145/3341105.3374025},
  booktitle = {Proceedings of the 35th Annual ACM Symposium on Applied Computing},
  pages = {892–899},
  numpages = {8},
  keywords = {deep learning, bias detection, hyperpartisan news, document representation, natural language processing},
  location = {Brno, Czech Republic},
  series = {SAC ’20}
}
```


## Usage

* train_nn.py
```
usage: train_nn.py [-h] [--news-dir DIR_PATH] [--ground-truth-dir DIR_PATH]
                   [--val-news-dir DIR_PATH] [--val-ground-truth-dir DIR_PATH]
                   [--test-news-dir DIR_PATH]
                   [--test-ground-truth-dir DIR_PATH] [--tensor-dataset PATH]
                   [--granularity {token,sentence,document,tokens_grouped_by_sentence}]
                   [--max-seq-len N] [--max-sent-len N]
                   [--embeddings-matrix-path EMBEDDINGS_MATRIX_PATH]
                   [--word2index-path WORD2INDEX_PATH]
                   [--embeddings-path EMBEDDINGS_PATH]
                   [--dataloader-workers N] [--epochs N] [--batch-size N]
                   [--learning-rate LR] [--seed N] [--resume PATH]
                   [--hidden-dim N] [--test-percentage TEST_PERCENTAGE]
                   [--reduce-lr-patience REDUCE_LR_PATIENCE] [--freeze FREEZE]
                   [--k-fold K] [--checkpoint-dir CHECKPOINT_DIR]
                   [--print-freq N] [--validate-every N]
                   [--checkpoint-every N]
                   name

positional arguments:
  name                  Model's name

optional arguments:
  -h, --help            show this help message and exit
  --news-dir DIR_PATH, --train-news-dir DIR_PATH
                        Hyperpartisan news XML directory (TRAIN)
  --ground-truth-dir DIR_PATH, --train-ground-truth-dir DIR_PATH
                        Ground truth XML directory (TRAIN)
  --val-news-dir DIR_PATH
                        Hyperpartisan news XML directory (VALIDATION)
  --val-ground-truth-dir DIR_PATH
                        Ground truth XML directory (VALIDATION)
  --test-news-dir DIR_PATH
                        Hyperpartisan news XML directory (TEST)
  --test-ground-truth-dir DIR_PATH
                        Ground truth XML directory (TEST)
  --tensor-dataset PATH
                        Path to a previously serialized dataset
  --granularity {token,sentence,document,tokens_grouped_by_sentence}
                        Granularity of embeddings in dataset
  --max-seq-len N       Maximum tokens to use for training (cutoff)
  --max-sent-len N      Maximum number of tokens in each sentence (cutoff)
  --embeddings-matrix-path EMBEDDINGS_MATRIX_PATH
                        Path to pre-generated embeddings matrix (needs
                        word2index)
  --word2index-path WORD2INDEX_PATH
                        Path to word-to-index mapping (corresponding to the
                        given embeddings matrix)
  --embeddings-path EMBEDDINGS_PATH
                        Path to pre-trained embeddings
  --dataloader-workers N
                        Number of workers to use for pytorch's Dataloader (0
                        means using main thread)
  --epochs N            Number of epochs to train for
  --batch-size N        Number of samples per batch (for training)
  --learning-rate LR    Number of epochs to train for
  --seed N              Random seed for initializing training
  --resume PATH         Resume training from the given checkpoint (default:
                        None)
  --hidden-dim N        Value for miscellaneous hidden dimensions
  --test-percentage TEST_PERCENTAGE
                        Percentage of samples to use for testing
  --reduce-lr-patience REDUCE_LR_PATIENCE
                        Number of non-improving epochs before reducing LR
  --freeze FREEZE       Indices of layers/sub-modules to freeze
  --k-fold K            Number of fold to use if using k-fold cross-validation
  --checkpoint-dir CHECKPOINT_DIR
                        Directory for saving model checkpoints
  --print-freq N        Frequency of epochs for printing train statistics
  --validate-every N    Frequency of evaluation on validation dataset (in
                        epochs)
  --checkpoint-every N  Checkpoint frequency (in epochs)

```


* serialize_pytorch_dataset.py
```
usage: serialize_pytorch_dataset.py [-h] --news-dir DIR_PATH
                                    [--ground-truth-dir DIR_PATH] --save-path
                                    PATH [--batch-size N]
                                    [--granularity {token,sentence,document,tokens_grouped_by_sentence}]
                                    [--max-seq-len N] [--max-sent-len N]
                                    [--embeddings-matrix-path EMBEDDINGS_MATRIX_PATH]
                                    [--word2index-path WORD2INDEX_PATH]
                                    [--embeddings-path EMBEDDINGS_PATH]
                                    [--dataloader-workers N]

optional arguments:
  -h, --help            show this help message and exit
  --news-dir DIR_PATH   Hyperpartisan news XML directory
  --ground-truth-dir DIR_PATH
                        Ground truth XML directory
  --save-path PATH      File path to save dataset on
  --batch-size N        Number of samples per batch (for training)
  --granularity {token,sentence,document,tokens_grouped_by_sentence}
                        Granularity of embeddings in dataset
  --max-seq-len N       Maximum tokens to use for training (cutoff)
  --max-sent-len N      Maximum number of tokens in each sentence (cutoff)
  --embeddings-matrix-path EMBEDDINGS_MATRIX_PATH
                        Path to pre-generated embeddings matrix (needs
                        word2index)
  --word2index-path WORD2INDEX_PATH
                        Path to word-to-index mapping (corresponding to the
                        given embeddings matrix)
  --embeddings-path EMBEDDINGS_PATH
                        Path to pre-trained embeddings
  --dataloader-workers N
                        Number of workers to use for pytorch's Dataloader (0
                        means using main thread)
```


* generate_token_embeddings.py
```
usage: generate_token_embeddings.py [-h] --articles-dir ARTICLES_DIR
                                    --embeddings-path EMBEDDINGS_PATH
                                    [--save-dir SAVE_DIR]
                                    [--base-name BASE_NAME]
generate_token_embeddings.py: error: the following arguments are required: --articles-dir, --embeddings-path
```

* evaluate_nn.py

```
usage: evaluate_nn.py [-h] --input-dir DIR_PATH [--output-dir DIR_PATH]
                      [--granularity {token,sentence,document,tokens_grouped_by_sentence}]
                      [--max-seq-len N] [--max-sent-len N]
                      [--embeddings-matrix-path EMBEDDINGS_MATRIX_PATH]
                      [--word2index-path WORD2INDEX_PATH]
                      [--embeddings-path EMBEDDINGS_PATH]
                      [--dataloader-workers N] [--model-path PATH]
                      [--batch-size N]
evaluate_nn.py: error: the following arguments are required: --input-dir

```

* train_kfold.py

## Scripts from the old sk-learn model

* generate_dataset.py
* grid_search.py
* evaluate.py
* train.py
