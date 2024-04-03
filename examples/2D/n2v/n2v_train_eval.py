"""
A script to train and evaluate a 2D N2V model.
"""
import argparse
import os
from pathlib import Path

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

def custom_mrc_reader(fpath, axes):
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[...,0].copy()


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

def train(datapath, traindir, just_eval=False,modelpath=None, poisson_noise_factor=-1, gaussian_noise_std=0.0):
    assert os.path.exists(datapath) #and os.path.isdir(datapath), f"Path {datapath} does not exist or is not a directory"
    # setting up the experiment.
    config = {'datapath':datapath, 'modelpath':modelpath, 'just_eval':just_eval}
    add_git_info(config)
    exp_directory = get_workdir(traindir, False)
    print(f"Experiment directory: {exp_directory}")
    print('')
    dump_config(config, exp_directory)
    hostname = socket.gethostname()
    wandb.init(name=os.path.join(hostname, *exp_directory.split('/')[-2:]), dir=traindir, project="N2V", config=config)

    data_type = "custom" if datapath.endswith(".mrc") else "tiff"
    mrc_read_fn = noise_gen_decorator(custom_mrc_reader, poisson_noise_factor=poisson_noise_factor, 
                                      gaussian_noise_std=gaussian_noise_std)
    
    read_source_func = mrc_read_fn if data_type == "custom" else None
    # loading/training the model and predicting
    model = get_model()
    if just_eval:
        assert modelpath is not None and os.path.exists(modelpath)
        model = model.load_from_checkpoint(modelpath)
    else: 
        train_data_module = CAREamicsTrainDataModule(
        train_path=datapath,
        val_path=datapath,
        data_type=data_type,
        read_source_func=read_source_func,
        patch_size=(64, 64),
        axes="SYX",
        batch_size=128,
        dataloader_params={"num_workers": 4},
        )
        trainer = Trainer(max_epochs=1, default_root_dir="bsd_test")
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
    )
    tiled_loop = CAREamicsFiring(trainer)
    trainer.predict_loop = tiled_loop
    preds = trainer.predict(model, datamodule=pred_data_module)
    preds = np.concatenate(preds, axis=0)
    outputpath = os.path.join(outputdir, os.path.basename(datapath).replace('.mrc', '.tif'))
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
    args = parser.parse_args()
    train(args.datapath, args.traindir, just_eval=args.just_eval, modelpath=args.modelpath, 
          poisson_noise_factor=args.poisson_noise_factor, gaussian_noise_std=args.gaussian_noise_std)

