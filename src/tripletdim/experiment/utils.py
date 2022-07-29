from pathlib import Path


def result_path(name, suffix, options):
    result_dir = Path("data") / name

    arg_name = "-".join(f"{key.replace('_', '')[:4]}_{value}" for key, value in sorted(options.items())
                        if value is not None).replace(".", "")
    result_file = (result_dir / arg_name).with_suffix(suffix)
    if not result_file.parent.exists():
        result_file.parent.mkdir(parents=True)
    return result_file


def result_paths(name, suffix, options):
    suffix_file = result_path(name, suffix, options)
    meta_file = result_path(name, '.meta.json', options)
    if suffix_file.exists() or meta_file.exists():
        raise RuntimeError(f"Result already exists in {suffix_file.stem}. Cancel run.")

    return meta_file, suffix_file