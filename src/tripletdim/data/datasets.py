import os
from pathlib import Path

import click
import numpy as np
import pandas
from cblearn import datasets
from cblearn import metrics
from cblearn import utils
from cblearn import preprocessing
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise
from sklearn.manifold import MDS
from scipy import special
from scipy import spatial
from scipy import io
import pandas as pd

from tripletdim.data.color import load_ekman_colors
from tripletdim.data.pitch import make_pitch_helix


## cblearn uses this directory to store datasets
os.environ['SCIKIT_LEARN_DATA'] = './data/raw'


def load_dataset(dataset, subject=None, random_state=None):
    objects, triplets = None, None

    ## ARTIFICIAL DATASETS
    if dataset.startswith('normal'):
        n_dimension, n_objects = map(int, dataset.split('-')[1:])
        objects = random_state.multivariate_normal(np.zeros(n_dimension), np.eye(n_dimension), size=(n_objects,))
        distance = 'euclidean'
    elif dataset == 'color-circle':
        similarity = load_ekman_colors().data
        objects = MDS(2, dissimilarity='precomputed', random_state=random_state).fit_transform(1 - similarity)
        distance = 'euclidean'
        n_dimension = 2
    elif dataset == 'pitch':
        objects = make_pitch_helix().data
        distance = 'euclidean'
        n_dimension = 3

    ## PSYCHOPHYSICAL DATASETS
    elif dataset == 'eidolon':
        # Siavash sent the Dataset via mail.
        triplets_per_subject = io.loadmat('data/raw/eidolon/Eidolon_triplets.mat')['triplets']
        triplets = triplets_per_subject[0, int(subject) - 1]  # allow subjects 1, 2, 3
        questions, answers = triplets[:, :3] - 1, triplets[:, 3]
        questions, answers = questions[answers != 0], answers[answers != 0]
        triplets = utils.check_query_response(questions, answers, result_format='list-order')
        n_dimension = 3
        distance = 'unknown'
    elif dataset == 'color-bostenboehm':
        def sample_rating_triplets(subject, random_state, n_triplets=500):
            data_root = Path('data/raw/bosten_boehm_2011_hue')
            df = pd.read_csv(data_root / f'{subject}11session1uniqueratingtable.txt', sep='\t',
                             index_col=False, names=['degree', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'RT'])
            target_colors = df.degree.unique()
            target_colors.sort()
            triplets = datasets.make_random_triplet_indices(size=n_triplets, n_objects=len(target_colors))

            ratings = df.set_index('degree').drop('RT', 'columns')
            rating_triplets = np.empty(shape=(*triplets.shape, 4))
            for row in range(len(triplets)):
                for col in range(3):
                    rating_triplets[row, col, :] = ratings.loc[target_colors[triplets[row, col]]].sample(
                        n=1, random_state=random_state)

            near = pairwise.paired_distances(rating_triplets[:, 0], rating_triplets[:, 1])
            far = pairwise.paired_distances(rating_triplets[:, 0], rating_triplets[:, 2])
            responses = near <= far
            return utils.check_query_response(triplets, responses, result_format='list-order')

        objects = [deg for deg in range(0, 360, 10)]
        triplets = sample_rating_triplets(subject, random_state)
        n_dimension = 2
        distance = 'euclidean'
    elif dataset == 'slant' or dataset == 'slant-2':
        path = Path('data/raw/slant_mlds_aguilar_wichmann_maertens_jov2017')
        file = path / f'before_gof_selection/O{subject}_full.csv'
        if not file.exists():  # always use the full dataset of 840 triplets
            file = path / f'O{subject}.csv'
        data = pandas.read_table(file, sep=' ')
        triplets = data.loc[:, ['i2', 'i1', 'i3']].values - 1
        response = data.Response.values.astype(bool)
        # which pair is more different?"
	    # 0 for pair (s1, s2), 1 for (s2, s3)
        # thus i1 is the anchor, and i2 is more similar if response is 1
        triplets = utils.check_query_response(triplets, response, result_format='list-order')
        n_dimension = 1
        distance = 'unknown'

    #### DATASETS NOT USED IN THE PAPER
    elif dataset == 'material':
        data = datasets.fetch_material_similarity(download_if_missing=False, random_state=random_state)
        triplets = data.triplet
        n_dimension = 9
        distance = 'unknown'
    elif dataset == 'imagenet':
        data = datasets.fetch_imagenet_similarity(version='0.1', download_if_missing=False, random_state=random_state)
        triplets = preprocessing.triplets_from_multiselect(data.data, 2, True)
        n_dimension = 9
        distance = 'unknown'
    elif dataset == 'transparency':
        data = [np.loadtxt(f'data/raw/transparency/sub1_block_{i}.txt', skiprows=1)
                for i in range(5)]
        data = np.row_stack(data)
        triplets_featured, responses = data[:, 2:8], data[:, 8].astype(bool)
        objects, triplets_flat = np.unique(np.vstack(np.split(triplets_featured, 3, axis=1)), axis=0, return_inverse=True)
        triplets = np.column_stack(np.split(triplets_flat, 3))
        triplets = utils.check_query_response(triplets, responses, result_format='list-order')
        n_dimension = 1
        distance = 'unknown'
    elif dataset == 'color-similarity':
        objects = 1 - load_ekman_colors().data
        distance = 'precomputed'
        n_dimension = 2
    elif dataset.startswith('corrnormal'):
        n_dimension, n_objects = map(int, dataset.split('-')[1:])
        cov_X = random_state.rand(n_dimension, n_dimension)
        objects = random_state.multivariate_normal(np.zeros(n_dimension), cov_X @ cov_X.T, size=(n_objects,))
        distance = 'euclidean'
    elif dataset.startswith('crowdnormal'):
        n_dimension, n_objects = map(int, dataset.split('-')[1:])
        objects = random_state.multivariate_normal(np.zeros(n_dimension), np.eye(n_dimension), size=(n_objects,))
        distance = 'euclidean'
        n_triplets = 4000
        n_subjects = int(subject)
        assert n_triplets % n_subjects == 0, f"Dataset expects that the triplets are dividable by the subjects, got {n_triplets} % {n_subjects} != 0"
        Ts = []
        Xs = []
        for subject_ix in range(n_subjects):
            X_subject = objects + np.random.normal(0, 0.2, objects.shape)
            T_subject = datasets.make_random_triplets(X_subject, size=n_triplets // n_subjects,
                                                      result_format='list-order', noise='normal',
                                                      noise_options={'scale': 0.3},
                                                      random_state=random_state)
            Ts.append(T_subject)
            Xs.append(X_subject)
        triplets = np.concatenate(Ts, axis=0)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    meta = {
        'name': dataset,
        'n_dimension': n_dimension,
        'distance': distance,
        'subject': subject
    }
    if objects is not None:
        meta['n_objects'] = len(objects)
    elif triplets is not None:
        meta['n_train_triplets'] = len(triplets)
    return meta, objects, triplets


def make_triplets(dataset, n_triplets=None, lambd=None, n_test_triplets=10000, noise_sd=0, subject='', random_state=None):
    rs = check_random_state(random_state)
    dataset_meta, objects, triplets = load_dataset(dataset, subject, rs)

    if objects is not None and triplets is None:
        n_objects, n_dimension = dataset_meta['n_objects'], dataset_meta['n_dimension']
        if n_triplets is not None and lambd is not None:
            raise ValueError("Only one of n_triplets and lambd must be specified.")
        elif n_triplets is None:
            if lambd is None:
                raise ValueError("Either n_triplets or lambd must be specified.")
            else:
                n_triplets = int(lambd * n_dimension * n_objects * np.log(n_objects))

        if dataset_meta['distance'] == 'euclidean':
            distance_matrix = pairwise.euclidean_distances(objects)
        else:
            distance_matrix = objects
        distance_vector = spatial.distance.squareform(distance_matrix, force='tovector', checks=False)
        distance_matrix /= np.std(distance_vector)

        options = dict(random_state=rs, result_format='list-order', noise='normal', noise_target='differences',
                       noise_options={'scale': noise_sd}, distance='precomputed')
        triplets = datasets.make_random_triplets(distance_matrix, size=n_triplets, **options)
        test_triplets = datasets.make_random_triplets(distance_matrix, size=n_test_triplets, **options)
        distance_differences = distance_matrix[triplets[:, [0, 1]]] - distance_matrix[triplets[:, [0, 2]]]
        n_all_triplets = (3 * special.comb(n_objects, 3))
        meta = {
            **dataset_meta,
            'n_train_triplets': n_triplets,
            'n_test_triplets': n_test_triplets,
            'frac_train_ndlogn': n_triplets / (n_dimension * n_objects * np.log(n_objects)),
            'frac_train_triplets': n_triplets / n_all_triplets,
            'max_train_score': metrics.query_accuracy(triplets, datasets.triplet_response(triplets, distance_matrix, distance='precomputed')),
            'max_test_score': metrics.query_accuracy(test_triplets, datasets.triplet_response(test_triplets, distance_matrix, distance='precomputed')),
            'noise_sd': noise_sd,
            'distance_sd': distance_matrix.std(),
            'distance_mean': distance_matrix.mean(),
            'difference_sd': distance_differences.std(),
            'difference_mean': distance_differences.mean(),
        }
    elif triplets is not None:
        test_triplets = None
        meta = {
            **dataset_meta,
            'n_train_triplets': len(triplets),
            'n_test_triplets': 0,
        }
    if isinstance(random_state, (int, str)):
        meta['random_state'] = random_state
    return meta, triplets, test_triplets


@click.command()
def fetch_datasets_cli(**kwargs):
    print("Download datasets if not already present. This may take a while.")
    print("Fetch material triplets ...")
    datasets.fetch_material_similarity()
    print("Fetch imagenet triplets ...")
    datasets.fetch_imagenet_similarity(version='0.1')
    print("Done.")


@click.command()
@click.option('--dataset', help='Name of the dataset')
@click.option('--n-triplets', type=int)
@click.option('--noise-sd', default=0, type=float)
def make_triplets_cli(**kwargs):
    meta, triplets, test_triplets = make_triplets(**kwargs)
    print(meta)


if __name__ == '__main__':
    make_triplets_cli()
