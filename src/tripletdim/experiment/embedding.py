import json

from sklearn.model_selection import ParameterGrid, GridSearchCV, RepeatedKFold
from sklearn import utils
import pandas as pd
from cblearn import embedding
import click

from tripletdim.data.datasets import make_triplets
from tripletdim.experiment.utils import result_paths


def repeated_gridsearch_crossval(model, train_data, test_data, model_config, random_state, n_repeats, n_splits):
    print(f"Repeated CV Gridsearch on {model.__class__.__name__}...")
    cv_folds = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state.randint(0, 2**32-1))
    cv_estimator = GridSearchCV(model, model_config, cv=cv_folds,
                                n_jobs=-1, refit=False, return_train_score=True, verbose=1)
    cv_estimator.fit(train_data)
    result = dict(cv_estimator.cv_results_)
    result.pop('params')
    if test_data is not None:
        result['train_score'] = []
        result['test_score'] = []
        print(f"Handoff test-set on {model.__class__.__name__}...")
        for param in ParameterGrid(model_config):
            model.set_params(**param)
            model.fit(train_data)
            result['train_score'].append(model.score(train_data))
            result['test_score'].append(model.score(test_data))
    return pd.DataFrame(result)


def embed_all_dimension(dataset, noise_sd, n_triplets=None, lambd=None, random_state=None, embedding_method='soe',
                        start_dimension=None, additional_dimension=2, subject=None, n_splits=10, n_repeats=10):
    rs = utils.check_random_state(random_state)
    if embedding_method == 'soe':
        estimator = embedding.SOE(random_state=rs, n_init=10)
    else:
        raise ValueError("Unknown model type")

    data_meta, train_triplets, test_triplets = make_triplets(dataset, n_triplets=n_triplets, lambd=lambd,
                                                             noise_sd=noise_sd, subject=subject, random_state=rs)
    if start_dimension is None:
        start_dimension = 1
    if start_dimension < 0:
        start_dimension += data_meta['n_dimension']
    embed_dimension = list(range(start_dimension, data_meta['n_dimension'] + additional_dimension + 1))
    config = {'n_components': embed_dimension}
    result = repeated_gridsearch_crossval(estimator, train_triplets, test_triplets, config, rs, n_repeats, n_splits)
    result.rename(columns={'param_n_components': 'embedding_dimension'}, inplace=True)

    meta = {'data': data_meta,
            'n_repeats': n_repeats,
            'n_splits': n_splits,
            'embedding_method': embedding_method,
            'embedding_dimension': embed_dimension}
    if isinstance(random_state, (int, str)):
        meta['random_state'] = random_state
    return meta, result


@click.command()
@click.option('--name', default="embedding", help='Name of the experiment')
@click.option('--dataset', help='Name of the dataset')
@click.option('--n_triplets', type=int)
@click.option('--lambd', type=float)
@click.option('--noise_sd', default=0, type=float)
@click.option('--random_state', default=None, type=int)
@click.option('--start_dimension', default=None, type=int)
@click.option('--additional_dimension', default=2, type=int)
@click.option('--subject', default=None, type=str)
@click.option('--n_repeats', default=10, type=int)
def embed_all_dimension_cli(name, **kwargs):
    meta_file, csv_file = result_paths(name, '.csv', kwargs)
    meta, result = embed_all_dimension(**kwargs)

    print(f"Save results to {csv_file}...")
    result.to_csv(csv_file)
    with meta_file.open("wt") as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    embed_all_dimension_cli()





