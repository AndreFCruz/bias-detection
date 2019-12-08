"""
Constructor functions for several Neural Network architectures.
"""

import torch
import torch.nn as nn

def construct_ffnn(input_dim, args=None):
    from architectures import FFNN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 150

    ffnn = FFNN(
        input_dim,
        hidden_layers=[hidden_dim, hidden_dim],
        output_dim=1, dropout=0.2,
        use_batch_norm=True
    )
    return nn.Sequential(
        nn.Dropout(0.4),      # Dropout on input embeddings
        ffnn
    )


def construct_cnn_bertha_von_suttner(input_dim, args=None):
    from architectures import TextCNN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 512
    cnn = TextCNN(
        input_dim,
        num_filters=hidden_dim,
        filter_sizes=[2, 3, 4, 5, 6],
        use_batch_norm=True
    )

    return nn.Sequential(
        # nn.Dropout2d(0.4),        ## Not in original architecture
        cnn,
        nn.Linear(cnn.output_dim(), 1),
        # nn.Dropout(0.2),          ## Not in original architecture
        nn.Sigmoid()
    )


def construct_cnn(input_dim, args=None):
    from architectures import FFNN, CNN, TextCNN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 10
    cnn = TextCNN(input_dim, hidden_dim)
    ffnn = FFNN(
        cnn.output_dim(),
        hidden_layers=[cnn.output_dim()],
        output_dim=1, dropout=0.2,
        use_batch_norm=True
    )
    return nn.Sequential(
        nn.Dropout2d(0.4),
        cnn,
        nn.Dropout(0.2),
        ffnn
    )


def construct_hierarch_att_net(input_dim, args=None, use_alternative_han=False):
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50
    from architectures import HierAttNet
    from architectures.han.hier_att_net_two import HierAttNetTwo
    if use_alternative_han:
        han = HierAttNetTwo(
            input_dim, hidden_dim, hidden_dim,
            args.max_seq_len, args.max_sent_len,
            dropout=0.2
        )
    else:
        han = HierAttNet(input_dim, hidden_dim, hidden_dim, dropout=0.2)
    return nn.Sequential(
        nn.Dropout2d(0.4),
        han
    )


def construct_multi_input(input_dim, args=None):
    from architectures.multi_input_arch import MultiInputHAN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50
    model = MultiInputHAN(input_dim, hidden_dim, hidden_dim)

    return nn.Sequential(
        nn.Dropout2d(0.4),
        model,
        nn.Dropout(0.2),
        nn.Linear(model.get_output_dim(), 1),
        nn.Sigmoid(),
    )


def construct_HAN_with_features(input_dim, args=None):
    from architectures.han.hier_att_net_for_doc_embeddings import HierAttNetForDocEmbeddings
    from architectures.mixed_architectures import EmbeddingsAndFeaturesNN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50

    han = HierAttNetForDocEmbeddings(input_dim, hidden_dim, hidden_dim, dropout=0.2)
    model = EmbeddingsAndFeaturesNN(han, hidden_dim * 2,
                num_features=54, embeddings_dropout=0.4)    ## NOTE Hardcoded number of features!
    return model


def construct_3HAN(input_dim, args=None):
    from architectures.han import ThreeHAN
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50
    model = ThreeHAN(input_dim, hidden_dim)

    return nn.Sequential(
        nn.Dropout2d(0.4),
        model
    )

def construct_lstm(input_dim, args=None):
    from architectures import LSTM
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50

    lstm = LSTM(input_dim, hidden_dim)
    return nn.Sequential(
        nn.Dropout(0.4),
        lstm,
        nn.Dropout(0.2),
        nn.Linear(hidden_dim * 2, 1),
        nn.Sigmoid()
    )

def construct_AttnBiLSTM(input_dim, args=None):
    hidden_dim = args.hidden_dim if (args and args.hidden_dim) else 50

    from architectures.han.sequence_encoder import LSTMSequenceEncoder
    attn_lstm = LSTMSequenceEncoder(
        input_dim, hidden_dim,
        bidirectional=True
    )
    return nn.Sequential(
        nn.Dropout(0.4),
        attn_lstm,
        nn.Dropout(0.2),
        nn.Linear(hidden_dim * 2, 1),
        nn.Sigmoid()
    )
