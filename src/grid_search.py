import pickle
import pprint
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV

def grid_search_params(estimator, param_grid, X, Y, name=""):
    # Setup Grid
    grid = GridSearchCV(
        estimator=estimator, param_grid=param_grid,
        cv=5, return_train_score=True,
        scoring='f1',
        n_jobs=-2, ## All CPUs but 1
    )

    # Train Models
    grid_results = grid.fit(X, Y)

    # Show Results
    print_grid_search_results(grid_results)

    # Save Best Model
    file_name = 'grid_search_best_' + name + ('_%.2f' % (grid.best_score_ * 100))
    save_object(grid.best_estimator_, file_name + '.pickle')
    return grid.best_estimator_


def print_grid_search_results(grid_results):
    print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def pprint_dict(d):
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(d)


if __name__ == "__main__":
    from train_sklearn import parse_args, load_dataset
    args = parse_args()

    X, y = load_dataset(args.dataset, args.undersampling)
    X = X.astype(np.float32)
    y = y.astype(np.float32).flatten()

    # Parameter Grid for Searching
    param_grid = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': np.arange(0.05, 0.5, 0.05),
        'n_estimators': [50, 100, 200, 300],
        'min_samples_split': [2, 3, 5, 10],
        'min_samples_leaf': [2, 3, 5, 10],
        'max_depth': [3, 4, 5, 6, 8, 10]
    }
    print("Grid Search Over: ")
    pprint_dict(param_grid)

    from sklearn.ensemble import GradientBoostingClassifier
    best_model = grid_search_params(
        GradientBoostingClassifier(),
        param_grid,
        X, y,
        name=args.name
    )
