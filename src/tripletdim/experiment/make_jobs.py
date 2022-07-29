import itertools
from pathlib import Path
import hashlib


def filter_records(records, filters):
    for key, value in filters.items():
        records = list(filter(lambda d: key in d and d[key] == value, records))
    return records

##############################################################
#### CHANGE THESE CONFIGS TO DEFINE EXPERIMENT CONDITIONS ####

## ARTIFICIAL DATASETS
simulation_datasets = [
    'normal-1-20', 'normal-2-20', 'normal-3-20', 
    'normal-1-60', 'normal-2-60', 'normal-3-60',
    'normal-8-60', 'normal-3-100', 'normal-8-100', 'color-circle', 'pitch',
]
simulation_noise_sd = [0.5, 1, 2]
simulation_lambdas = [2, 4, 8]
simulation_configs = [{'dataset': k, 'lambd': l, 'noise_sd': n}
                      for k in simulation_datasets
                      for l in simulation_lambdas
                      for n in simulation_noise_sd]

## PSYCHOPHYSICAL DATASETS
psy_configs = [{'dataset': 'eidolon', 'subject': '1'},
               {'dataset': 'eidolon', 'subject': '2'},
               {'dataset': 'eidolon', 'subject': '3'},
               {'dataset': 'color-bostenboehm', 'subject': 'WL1'},
               {'dataset': 'color-bostenboehm', 'subject': 'VH1'},
               {'dataset': 'color-bostenboehm', 'subject': 'KM1'},
               {'dataset': 'material'},
               {'dataset': 'imagenet', 'n_repeats': 1},
               {'dataset': 'slant', 'subject': 1},
               {'dataset': 'slant', 'subject': 2},
               {'dataset': 'slant', 'subject': 3},
               {'dataset': 'slant', 'subject': 4},
               {'dataset': 'slant-2', 'subject': 5},
               {'dataset': 'slant-2', 'subject': 6},
               {'dataset': 'slant-2', 'subject': 7},
               {'dataset': 'slant-2', 'subject': 8},]


## ERROR ANALYSIS EXPERIMENTS
hypotest_analysis_configs  = [
    {'name': f"hypotest/{record['dataset']}",
     'alpha': 0.05,
     'note': f'rep{repeat}', **record}
    for record in [
    *filter_records(simulation_configs, {'dataset': 'normal-1-20'}),
    *filter_records(simulation_configs, {'dataset': 'normal-1-60'}),
    *filter_records(simulation_configs, {'dataset': 'normal-3-60'}),
    *filter_records(simulation_configs, {'dataset': 'normal-8-100'}),
    *filter_records(psy_configs, {'dataset': 'eidolon', 'subject': '1'})
    ]
    for repeat in range(100)
]

####                                                      ####
##############################################################


def jobs_from_configs(command, configs, seed):
    arguments = (" ".join(f"--{k}='{v}'" for k, v in config.items()) for config in configs)
    commands = [f"{command} {args}" for args in arguments]
    # seed random state with hash from command and global seed
    # to ensure reproducibility.
    # Changing the command will change the seed, but otherwise the seed will stay constant.
    hashs = (hashlib.sha1((str(seed) + command).encode()) for command in commands)
    seeds = [int.from_bytes(command_hash.digest(), 'big') % 2 ** 32
             for command_hash in hashs]
    return [f"{command} --random_state='{seed}'\n" for command, seed in zip(commands, seeds)]


def save_jobs(job_name, command, configs, seed):
    jobs = jobs_from_configs(command, configs, seed)
    job_file = Path(f'jobs/{job_name}.txt').absolute()
    with job_file.open('wt') as f:
        f.writelines(jobs)


def main():
    save_jobs("simulations-embedding", "embedding", simulation_configs, seed=42)
    save_jobs("psy-embedding", "embedding", psy_configs, seed=42)
    for name, configs in itertools.groupby(hypotest_analysis_configs, lambda x: x['dataset']):
        save_jobs(f"hypotest-{name}", "hypotest", configs, seed=42)
    #save_jobs("hypotest-error", "hypotest-error", hypotest_analysis_configs, seed=42)




