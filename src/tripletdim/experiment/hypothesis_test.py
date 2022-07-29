import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest

from tripletdim.experiment import embedding
from tripletdim.experiment.utils import result_paths


def sequential_crossval_ttest(sample_sequence, n_splits, alpha):
    """
    sample_sequence: Array of shape (n_samples, n_steps)
    """
    differences = np.diff(sample_sequence)
    n_samples, n_steps = differences.shape

    test_train_ratio = 1 / (n_splits - 1)
    effect = differences.mean(axis=0) / differences.std(axis=0, ddof=1)
    # Nadeau and Bengio correction of dependent Student's t-test due to Cross Validation
    t_stats = effect / np.sqrt(1 / n_samples + test_train_ratio)

    p_values = stats.t.sf(t_stats, n_samples - 1)  # right-tailed t-test
    assert t_stats.shape == (n_steps,)

    # holm-bonferroni correction
    result = multitest.multipletests(p_values, alpha=alpha, method='holm')
    return {'reject': result[0], 'pvals': p_values, 'pvals_corrected': result[1],
            'tstats': t_stats, 'effectsize': effect,
            'alpha': alpha, 'alpha_corrected': result[3], 'step': np.arange(n_steps)}


def measure_sequential_test_error(correct_reject, test_results):
    type1 = 0  # reject null hypothesis, but null hypothesis is true
    type2 = 0  # accept null hypothesis, but null hypothesis is false
    fwer = 0
    for test_result in test_results:
        pred_reject = np.array(test_result['reject'], dtype=bool)
        mistake = (correct_reject != pred_reject)
        type1 += mistake & pred_reject
        type2 += mistake & ~pred_reject
        fwer += any(mistake & pred_reject)
    type1 = type1 / len(test_results)
    type2 = type2 / len(test_results)
    meta = {
        'correct_reject': correct_reject.tolist(),
        'fwer': fwer,
        'type1_error': type1.tolist(),
        'type2_error': type2.tolist(),
        'power': (1 - type2).tolist()
    }
    return meta, pd.concat(pd.DataFrame(r) for r in test_results)


def measure_seq_crossval_ttest_error(correct_reject, data, cv_folds, alpha):
    test_results = [sequential_crossval_ttest(d, cv_folds, alpha) for d in data]
    return measure_sequential_test_error(correct_reject, test_results, alpha)


def dimension_hypothesis_test(score_df, alpha=0.05, multitest='holm'):
    cv_score_df = score_df\
        .set_index(["embedding_dimension"])\
        .sort_index()\
        .filter(regex=r"split[0-9]+_test_score", axis="columns")
    n_splits = len(cv_score_df.columns)
    test_result = sequential_crossval_ttest(cv_score_df.values.T, n_splits, alpha)
    test_result['dimension_step'] = [f'{first}-{second}' for (first, second) in
                                     zip(cv_score_df.index.unique()[:-1], cv_score_df.index.unique()[1:])]
    return test_result


def embed_dimension_and_hypothesis_test(dataset, noise_sd, random_state=None, alpha=0.05, **kwargs):
    meta, score_df = embedding.embed_all_dimension(dataset, noise_sd=noise_sd, random_state=random_state, **kwargs)
    meta = {**meta, 'alpha': alpha}
    return meta, dimension_hypothesis_test(score_df, alpha=alpha)


def measure_dimension_test_error(meta_df, test_results):
    assert len(pd.unique(meta_df['data.name'])) == 1
    assert len(pd.unique(meta_df['data.n_dimension'])) == 1
    assert len(pd.unique(meta_df['data.noise_sd'])) == 1
    assert len(pd.unique(meta_df['data.n_train_triplets'])) == 1
    dataset = meta_df['data.name'].iloc[0]
    data_dimension = meta_df['data.n_dimension'].iloc[0]
    n_embedding_steps = len(meta_df['embedding_dimension'].iloc[0])
    n_test_repetitions = len(test_results)

    correct_reject = np.full(n_embedding_steps - 1, False)
    correct_reject[:(data_dimension - 1)] = True

    meta_test, df = measure_sequential_test_error(correct_reject, test_results)
    meta = {
        **meta_df.to_dict(orient='records')[0],
        **meta_test,
        'dataset': dataset,  # for backwards compatibility, this should be same as data.name
        'n_test_repetition': n_test_repetitions,
        'test_steps': list(range(1, len(correct_reject) + 1)),
    }
    return meta, df


@click.command()
@click.option('--name', default="hypotest", help="Name of the experiment")
@click.option('--dataset', help='Name of the dataset')
@click.option('--alpha', type=float)
@click.option('--lambd', type=int)
@click.option('--n_triplets', type=int)
@click.option('--noise_sd', default=0, type=float)
@click.option('--random_state', default=None, type=int)
@click.option('--note', default=None, help='Another flag to note down information')
def dimension_hypothesis_test_cli(name, **kwargs):
    meta_file, csv_file = result_paths(name, '.csv', kwargs)
    del kwargs['note']
    print(kwargs)
    meta, result = embed_dimension_and_hypothesis_test(**kwargs)

    print(f"Save results to {csv_file}...")
    pd.DataFrame(result).to_csv(csv_file)
    with meta_file.open("wt") as f:
        json.dump(meta, f, indent=2)

@click.command()
@click.option('--name', default="hypotest-error", help="Name of the experiment")
@click.option('--result_dir', type=str)
def measure_test_error_cli(name, result_dir, **kwargs):
    result_dir = Path(result_dir)
    test_metas = pd.concat({f: pd.json_normalize(json.loads(f.read_text())) for f in result_dir.glob('*.meta.json')})

    for (noise_sd, n_triplets), meta_group in test_metas.groupby(['data.noise_sd', 'data.n_train_triplets']):
        data_files = (f.with_suffix('').with_suffix('.csv') for f in meta_group.index.get_level_values(0))
        test_results = [pd.read_csv(f, index_col=False).to_dict(orient='list') for f in data_files]
        print(len(test_results))
        meta_file, csv_file = result_paths(name, '.csv', {**kwargs, 'triplets': n_triplets, 'noise': noise_sd})
        meta, result = measure_dimension_test_error(meta_group, test_results)

        print(f"Save results to {csv_file}...")
        result.to_csv(csv_file)
        with meta_file.open("wt") as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    measure_test_error_cli()




