"""
A script to train and evaluate a 2D N2V model.
"""
import argparse
import os
from pathlib import Path
import json

import torch
import albumentations as Aug
import socket
import matplotlib.pyplot as plt
import tifffile
from careamics_portfolio import PortfolioManager
from pytorch_lightning import Trainer
import wandb
import git
from skimage.io import imsave as imsave_skimage
from skimage.io import imread

from careamics import CAREamicsModule
from careamics.lightning_prediction import CAREamicsFiring
from careamics.ligthning_datamodule import (
    CAREamicsPredictDataModule,
    CAREamicsTrainDataModule,
)
from careamics.transforms import N2VManipulate
from careamics.utils.metrics import psnr
from careamics.utils.experiment_saving import dump_config, add_git_info, get_workdir
from read_mrc import read_mrc  
import numpy as np
from read_czi import load_data as load_czi_data
from careamics.dataset.dataset_utils import list_files
from read_nd2 import load_one_file as load_nd2_data


def noise_gen_decorator(data_func, poisson_noise_factor=-1, gaussian_noise_std=0.0):
    def wrapper(*args, **kwargs):
        noisy_data = data_func(*args, **kwargs)
        if poisson_noise_factor > 0:
            print('Enabling poisson noise with factor', poisson_noise_factor)
            # The higher this factor, the more the poisson noise.
            noisy_data = np.random.poisson(noisy_data / poisson_noise_factor)

        if gaussian_noise_std > 0.0:
            print('Adding gaussian noise', gaussian_noise_std)
            noisy_data = noisy_data + np.random.normal(0, gaussian_noise_std, noisy_data.shape)
        return noisy_data
    return wrapper

def load_tiff(path, axes=None):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin='tifffile')
    return data

def custom_mrc_reader(fpath, axes):
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[...,0].copy()

def get_extension(fpath):
    assert os.path.exists(fpath)
    if os.path.isdir(fpath):
        extensions = list(map(lambda x: x.split('.')[-1], os.listdir(fpath)))
        assert len(extensions) > 0, f"No files found in {fpath}"
        assert len(set(extensions)) == 1, f"Multiple extensions found: {set(extensions)}"
        ext = extensions[0]
    else:
        ext = fpath.split('.')[-1]
    
    print('Working with extension:', ext)
    return ext

def select_channels(data_load_fn, channel_idx, channel_dim):
    """
    Args:
        data_load_fn: A function that loads the data.
        channel_idx: The index of the channel to select.
        channel_dim: The dimension of the channel.
    """
    def wrapper(*args, **kwargs):
        data = data_load_fn(*args, **kwargs)
        if channel_dim == 0:
            return data[channel_idx].copy()
        elif channel_dim == 1:
            return data[:, channel_idx].copy()
        elif channel_dim == 2:
            return data[:, :, channel_idx].copy()
        elif channel_dim == 3:
            return data[:, :, :, channel_idx].copy()
    return wrapper


def get_model():
    model = CAREamicsModule(
    algorithm="n2v",
    loss="n2v",
    architecture="UNet",
    model_parameters={"n2v2": True},
    optimizer_parameters={"lr": 1e-3},
    lr_scheduler_parameters={"factor": 0.5, "patience": 10},
    )
    return model

def get_data_type(datapath, channel_idx,channel_dim, extension):
    if datapath.endswith(".mrc") or extension == 'mrc':
        return "custom"
    elif datapath.endswith('.czi') or extension == 'czi':
        return "custom"
    elif channel_idx is not None:
        assert channel_dim is not None and isinstance(channel_dim, int)
        return "custom"
    else:
        return "tiff"

def get_output_path(outputdir, datapath):
    if os.path.isdir(datapath):
        datapath = datapath.strip('/')
        fname = os.path.basename(datapath)
    else:
        assert os.path.exists(datapath)  
        fname = os.path.basename(datapath)
        # replace any extension with tif
        fname = '.'.join(fname.split('.')[:-1])

    assert fname != '', f"Could not extract filename from {datapath}"
    fname += '.tif'
    return os.path.join(outputdir, fname)

def get_read_source_func(extension,channel_idx, channel_dim, poisson_noise_factor, gaussian_noise_std, data_type):
    if extension == 'mrc':
        read_source_func = noise_gen_decorator(custom_mrc_reader, poisson_noise_factor=poisson_noise_factor, 
                                        gaussian_noise_std=gaussian_noise_std)
    elif extension == 'tif':
        read_source_func = noise_gen_decorator(load_tiff, poisson_noise_factor=poisson_noise_factor,
                                        gaussian_noise_std=gaussian_noise_std)
    elif extension == 'czi':
        read_source_func = noise_gen_decorator(load_czi_data, poisson_noise_factor=poisson_noise_factor,
                                        gaussian_noise_std=gaussian_noise_std)
    elif extension == 'nd2':
        read_source_func = noise_gen_decorator(load_nd2_data, poisson_noise_factor=poisson_noise_factor,
                                        gaussian_noise_std=gaussian_noise_std)
    
    if channel_idx is not None:
        read_source_func = select_channels(read_source_func, channel_idx, channel_dim)
        assert data_type == "custom"
    
    if data_type != "custom":
        read_source_func = None

    return read_source_func

def train(datapath, traindir, just_eval=False,modelpath=None, poisson_noise_factor=-1, gaussian_noise_std=0.0, max_epochs=100,
          channel_idx=None, channel_dim=None):
    assert os.path.exists(datapath) #and os.path.isdir(datapath), f"Path {datapath} does not exist or is not a directory"
    # setting up the experiment.
    config = {'datapath':datapath, 'modelpath':modelpath, 'just_eval':just_eval, 'poisson_noise_factor':poisson_noise_factor,
              'gaussian_noise_std':gaussian_noise_std, 'max_epochs':max_epochs, 'channel_idx':channel_idx, 'channel_dim':channel_dim}
    add_git_info(config)
    exp_directory = get_workdir(traindir, False)
    print(f"Experiment directory: {exp_directory}")
    print('')
    dump_config(config, exp_directory)
    hostname = socket.gethostname()
    wandb.init(name=os.path.join(hostname, *exp_directory.split('/')[-2:]), dir=traindir, project="N2V", config=config)


    extension = get_extension(datapath)
    data_type = get_data_type(datapath, channel_idx, channel_dim, extension)
    read_source_func = get_read_source_func(extension,channel_idx, channel_dim, poisson_noise_factor, gaussian_noise_std, data_type)
    
    # loading/training the model and predicting
    model = get_model()
    if just_eval:
        assert modelpath is not None and os.path.exists(modelpath)
        model = torch.load(modelpath)
        config_fpath = os.path.join(os.path.dirname(modelpath), 'config.json')
        with open(config_fpath, 'r') as f:
            config = json.load(f)
            assert config['channel_idx'] == channel_idx, f"Expected {channel_idx} got {config['channel_idx']}"
            assert config['channel_dim'] == channel_dim, f"Expected {channel_dim} got {config['channel_dim']}"
            assert config['poisson_noise_factor'] == poisson_noise_factor, f"Expected {poisson_noise_factor} got {config['poisson_noise_factor']}"
            assert config['gaussian_noise_std'] == gaussian_noise_std, f"Expected {gaussian_noise_std} got {config['gaussian_noise_std']}"
            if os.path.isdir(datapath):
                assert config['datapath'].strip('/') == datapath.strip('/'), f"Expected {datapath} got {config['datapath']}"
                # assert config['datapath'] == datapath, f"Expected {datapath} got {config['datapath']}"
        trainer = Trainer()
    else: 
        train_data_module = CAREamicsTrainDataModule(
        train_path=datapath,
        val_path=None,
        data_type=data_type,
        read_source_func=read_source_func,
        patch_size=(64, 64),
        axes="SYX",
        batch_size=128,
        dataloader_params={"num_workers": 4},
        )
        trainer = Trainer(max_epochs=max_epochs, default_root_dir="bsd_test")
        trainer.fit(model, datamodule=train_data_module)

    if just_eval != True:
        torch.save(model, os.path.join(exp_directory, "last_model.net"))

    
    outputdir = exp_directory
    pred_data_module = CAREamicsPredictDataModule(
    pred_path=datapath,
    data_type=data_type,
    read_source_func=read_source_func,
    tile_size=(256, 256),
    axes="SYX",
    batch_size=1,
    tta_transforms=True,
    dataloader_params={"num_workers": 4},
    )
    tiled_loop = CAREamicsFiring(trainer)
    trainer.predict_loop = tiled_loop
    preds = trainer.predict(model, datamodule=pred_data_module)
    preds = np.concatenate(preds, axis=0)
    outputpath = get_output_path(outputdir, datapath)
    print('Saving predictions to:', outputpath, ' data shape:', preds.shape)
    imsave_skimage(outputpath, preds, plugin='tifffile')
    # save preds
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--just_eval', action='store_true')
    parser.add_argument('--traindir', type=str, default=os.path.expanduser('~/training/N2V/'))
    parser.add_argument('--gaussian_noise_std', type=float, default=0.0)
    parser.add_argument('--poisson_noise_factor', type=float, default=-1)
    parser.add_argument('--max_epochs', type=int, default=400)
    parser.add_argument('--channel_idx', type=int, default=None)
    parser.add_argument('--channel_dim', type=int, default=None)
    args = parser.parse_args()
    # list(Path('/group/jug/ashesh/data/expansion_microscopy_v2/').glob('*.czi'))
    train(args.datapath, args.traindir, just_eval=args.just_eval, modelpath=args.modelpath, 
          poisson_noise_factor=args.poisson_noise_factor, gaussian_noise_std=args.gaussian_noise_std,
          max_epochs=args.max_epochs, channel_idx=args.channel_idx, channel_dim=args.channel_dim)

