import json
import os
import time
from datetime import datetime
from pathlib import Path

import git


def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(f'Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed')
            exit()
    if len(versions) == 0:
        return '0'
    return f'{max(versions) + 1}'


def get_month():
    return datetime.now().strftime("%y%m")


def load_config(directory):
    config_fpath = os.path.join(directory, 'config.json')
    if not os.path.exists(config_fpath):
        return None

    with open(config_fpath, 'r') as f:
        return json.load(f)


def dump_config(config, directory):
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(config, f)


def get_workdir(root_dir, use_max_version, nested_call=0):
    rel_path = get_month()
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f'{version - 1}'

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f'Workdir {cur_workdir} already exists. Probably because someother program also created the exact same directory. Trying to get a new version.'
        )
        time.sleep(2.5)
        if nested_call > 10:
            raise ValueError(f'Cannot create a new directory. {cur_workdir} already exists.')

        return get_workdir(root_dir, use_max_version, nested_call + 1)
    return cur_workdir


def add_git_info(config):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(dir_path, search_parent_directories=True)
    config['git'] = {}
    config['git']['changedFiles'] = [item.a_path for item in repo.index.diff(None)]
    config['git']['branch'] = repo.active_branch.name
    config['git']['untracked_files'] = repo.untracked_files
    config['git']['latest_commit'] = repo.head.object.hexsha
