import numpy as np
from scipy import stats

from tripletdim.experiment import hypothesis_test


cv_folds = 10
means = [1, 2, 3, 3.001, 2]
correct_reject = [True, True, False, False]


def sample_sequence(seed=None):
    n_samples = 100
    n_reps = 10
    rs = np.random.RandomState(seed)
    data = stats.norm.rvs(size=n_samples, loc=0, scale=1, random_state=rs)
    repeated_data = np.asarray([rs.permutation(data) for i in range(n_reps)])
    step_samples = np.asarray([repeated_data + stats.norm.rvs(size=n_samples, loc=m, scale=0.1, random_state=rs)
                               for m in means])
    cv_step_samples = np.asarray([np.split(d, cv_folds, axis=-1) for d in step_samples])
    sample_sequence = [[d[r][np.arange(cv_folds) != cv].mean() for cv in range(cv_folds) for r in range(n_reps)]
                       for d in cv_step_samples]
    return np.asarray(sample_sequence)


def test_sequential_crossval_ttest():
    result = hypothesis_test.sequential_crossval_ttest(sample_sequence(), cv_folds, alpha=0.05)
    assert (result['reject'] == correct_reject).all()
    assert (np.log10(result['pvals_corrected'][0:2]) < -10).all()
    assert (np.log10(result['pvals_corrected'][2]) > -2)
    assert (result['pvals_corrected'][3] == 1)


def test_measure_seq_crossval_ttest_error():
    datasets = [sample_sequence(i) for i in range(100)]
    for alpha in [0.01, 0.05, 0.1]:
        type1, type2 = hypothesis_test.measure_seq_crossval_ttest_error(correct_reject, datasets, cv_folds, alpha)
        assert np.allclose(type1, [0, 0, alpha, 0], rtol=1)
        assert np.allclose(type2, [0, 0, 0, 0])
