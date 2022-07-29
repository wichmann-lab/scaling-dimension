from setuptools import find_packages, setup

setup(
    name='tripletdim',
    version='0',
    packages=find_packages(),
    package_dir={'': 'src'},
    entry_points = {
        'console_scripts': [
            'embedding=tripletdim.experiment.embedding:embed_all_dimension_cli',
            'hypotest=tripletdim.experiment.hypothesis_test:dimension_hypothesis_test_cli',
            'hypotest-error=tripletdim.experiment.hypothesis_test:measure_test_error_cli',
            'make-jobs=tripletdim.experiment.make_jobs:main',
            'fetch-datasets=tripletdim.data.datasets:fetch_datasets_cli',
            'plot-simulations=tripletdim.plot.plot_simulations:main',
            'plot-tripletscore=tripletdim.plot.plot_triplet_score:main',
        ]
    },
    url='',
    license='GPLv3',
    author='David-Elias Künstle',
    author_email='david-elias.kuenstle@uni-tuebingen.de',
    description='Experiment code for estimating the dimension from triplets'
)
