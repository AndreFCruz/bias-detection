#!/usr/bin/env python3

"""
Train script
"""

import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--numpy-dataset', dest='dataset',
                        help='Path to a previously serialized dataset',
                        required=True, type=str, metavar='PATH')

    # parser.add_argument('--test-dataset', dest='test_dataset',
    #                     help='Path to a previously serialized dataset (Test)',
    #                     required=True, type=str, metavar='PATH')

    parser.add_argument('--undersampling',
                        help='Whether to use undersampling to balance classes in the dataset',
                        action='store_true')

    parser.add_argument('name', type=str, help='Classifier\'s name')

    args = parser.parse_args()
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('\n')

    return args


def load_dataset(path, undersampling=False):
    dataset = np.load(path)
    X, Y = dataset['X'], dataset['Y']

    if undersampling:
        from nn_utils import balanced_sampling
        balanced_indices = balanced_sampling(Y)
        X, Y = X[balanced_indices], Y[balanced_indices]

    return X, Y


def train_sklearn_classifier(classifier, X, y):
    print('Cross-validating')
    cv = cross_validate(
        classifier, X, y, cv=5, return_train_score=True,
        scoring=['accuracy', 'precision', 'recall', 'f1']
    )

    for metric, scores in sorted(cv.items()):
        print('{} : {:.3%} +- {:.2%}'.format(metric, np.mean(scores), np.std(scores)))
        print('->', scores, end='\n\n')

    classifier.fit(X, y)
    return classifier


def main_semeval():
    args = parse_args()

    X, y = load_dataset(args.dataset, undersampling=args.undersampling)
    X = X.astype(np.float32)
    y = y.astype(np.float32).flatten()

    ## Train sk-learn classifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    classifier = train_sklearn_classifier(
        # RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(
            n_estimators=100, min_samples_leaf=10, loss='exponential', learning_rate=0.1, min_samples_split=2, max_depth=8),
            # n_estimators=100, min_samples_leaf=10, loss='deviance', learning_rate=0.1, min_samples_split=2, max_depth=10),
            # n_estimators=200, min_samples_leaf=10, loss='exponential', learning_rate=0.05, min_samples_split=10, max_depth=10),
            # n_estimators=300, min_samples_leaf=5, loss='deviance', learning_rate=0.4, min_samples_split=5, max_depth=5),
        X, y
    )

    ## Save classifier
    import pickle, random
    pickle.dump(classifier, open('checkpoints/sklearn-clf.{}.pickle'.format(args.name + str(random.randint(0, 100))), 'wb'))


if __name__ == '__main__':
    main_semeval()
