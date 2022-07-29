#%%
import itertools

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from cblearn.embedding import SOE
from scipy.spatial import procrustes

from tripletdim.data.color import load_ekman_colors
from tripletdim.data.pitch import make_pitch_helix
from tripletdim.data.datasets import make_triplets

#%%
def _plot_color_dataset(embedding, annotate=False, **kwargs):
    fig = plt.gcf()
    ekman = load_ekman_colors()
    n_dim = embedding.shape[1]
    if n_dim == 1:
        plt.gca().get_yaxis().set_visible(False)
        embedding = np.c_[embedding, np.zeros(embedding.size)]
        
    if n_dim in (1, 2):
        ax = plt.gca()
        ax.scatter(embedding[:, 0], embedding[:, 1], c=ekman.color, **kwargs)
    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.axes.zaxis.set_ticklabels([]) 
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.5, 1]))
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=40, c=ekman.color, **kwargs)
    else:
        raise ValueError()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([]) 
    
    if annotate:
        for coords, label in zip(embedding, ekman.feature_names):
            ax.text(*coords, '   ' + label, fontsize=10, va='center')
    return ax


def plot_color_dataset_mds(n_dim=2, annotate=False):
    ekman = load_ekman_colors()
    embedding = MDS(n_dim, dissimilarity='precomputed', random_state=48).fit_transform(1 - ekman.data)
    _plot_color_dataset(embedding, annotate)


def plot_color_dataset_soe(n_triplets, noise_sd, n_repeat, show_disparity=False, annotate=False, color=None, **kwargs):
    embeddings = []
    scores, train_scores = [], []
    ekman = load_ekman_colors()
    anker_embedding = MDS(2, dissimilarity='precomputed', random_state=48).fit_transform(1 - ekman.data)
    soe = SOE(2)
    for _ in range(n_repeat):
        _, triplets, test_triplets = make_triplets('color-circle', int(n_triplets), noise_sd=float(noise_sd))
        embedding = soe.fit_transform(triplets)
        embeddings.append(embedding)
        train_scores.append(soe.score(triplets))
        scores.append(soe.score(test_triplets))
    disparities = []
    for embedding in embeddings:
        _, embedding, disparity = procrustes(anker_embedding, embedding)
        disparities.append(disparity)
        ax = _plot_color_dataset(embedding, annotate, **kwargs)
    ax.text(0.5, 0.97, f"accuracy = ${np.mean(scores):.3f}$", ha='center', va='top', transform=ax.transAxes)
    if show_disparity:
        ax.text(0.5, 0.12, f"train - test = ${np.mean(train_scores) - np.mean(scores):.2f}$", ha='center', va='bottom',
                transform=ax.transAxes)
        ax.text(0.5, 0.02, f'RMSE = ${np.mean(np.sqrt(disparities)):.2f}$', ha='center', va='bottom', transform=ax.transAxes)


def plot_color_grid_soe(triplets, noises, **kwargs):
    df = pd.DataFrame(list(itertools.product(triplets, noises)), columns=['triplets', 'noise'])
    g = sns.FacetGrid(df, col='triplets', row='noise', row_order=noises, margin_titles=True)

    g.map(plot_color_dataset_soe, 'triplets', 'noise', n_repeat=50, alpha=0.1, **kwargs)
    g.set_titles(col_template="#Triplets = {col_name}", row_template='Noise SD = {row_name}')
    g.set_axis_labels("", "")
    g.despine(top=False, right=False)
    g.tight_layout()
    # for ax in g.axes.ravel():
    #     ax.set_frame_on(False)
    return g


def plot_pitch_dataset():
    pitch = make_pitch_helix(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.5, 1]))

    ax.plot(pitch.data[:, 0], pitch.data[:, 1], pitch.data[:, 2], marker='o')
    ax.set_xlabel('Chroma')
    ax.set_ylabel('Chroma')
    ax.set_zlabel('Height')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.view_init(25, 65)
    for coord, label in zip(pitch.data, pitch.helmholz):
        ax.text(*coord.T, '  ' + label, fontsize=10, ha='left', va='center')


def main():
    sns.set_theme('talk', 'ticks')

    print("Pitch 3d")
    plot_pitch_dataset()
    plt.tight_layout()
    sns.despine()
    plt.savefig('tex/plots/pitch-helix.pdf')
    plt.close()

    with sns.axes_style('white'):
        print(f"Large Ekman Grid")
        noises = [2, 1, 0.5, 0]
        triplets = [125, 250, 500, 1000, 2000]
        g = plot_color_grid_soe(triplets, noises, show_disparity=True)
        g.savefig(f'tex/plots/ekman-colors-grid-large.pdf', bbox_inches='tight')
        plt.close()

        print(f"Small Ekman Grid")
        g = plot_color_grid_soe([500, 1000, 2000], [2, 0.5])
        g.savefig(f'tex/plots/ekman-colors-grid-small.pdf', bbox_inches='tight')
        plt.close()
