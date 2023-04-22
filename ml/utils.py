from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from ml.model import CNN, ResNet, SAE, MLP

SEED = 9876


def train_mlp(
    output_dim,
    data_path,
    epoch,
    model_path,
    signal_length,
    logger,
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=SEED, workers=True)

    model = MLP(
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(
        val_check_interval=1.0,
        max_epochs=epoch,
        devices="auto",
        accelerator="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="training_loss", mode="min", check_on_train_epoch_end=True
            )
        ],
    )
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_sae(
    output_dim,
    data_path,
    epoch,
    model_path,
    signal_length,
    logger
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=SEED, workers=True)

    model = SAE(
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    
    # train ae1
    for train_ae_idx in [1,2,3,4,5,0]:
        print('training ae_idx: ', train_ae_idx)
        model.train_ae_idx = train_ae_idx
        trainer = Trainer(
            val_check_interval=1.0,
            max_epochs=epoch,
            devices="auto",
            accelerator="auto",
            logger=logger,
            callbacks=[
                EarlyStopping(
                    monitor="training_loss", mode="min", check_on_train_epoch_end=True
                )
            ],
        )
        trainer.fit(model)
        print('training FINISHED on ae_idx: ', train_ae_idx)
        # save model
        trainer.save_checkpoint(str(model_path.absolute()))


def train_cnn(
    c1_kernel_size,
    c1_output_dim,
    c1_stride,
    c2_kernel_size,
    c2_output_dim,
    c2_stride,
    output_dim,
    data_path,
    epoch,
    model_path,
    signal_length,
    logger,
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=SEED, workers=True)

    model = CNN(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c2_kernel_size=c2_kernel_size,
        c2_output_dim=c2_output_dim,
        c2_stride=c2_stride,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(
        val_check_interval=1.0,
        max_epochs=epoch,
        devices="auto",
        accelerator="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="training_loss", mode="min", check_on_train_epoch_end=True
            )
        ],
    )
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_resnet(
    c1_kernel_size,
    c1_output_dim,
    c1_stride,
    c1_groups,
    c1_n_block,
    output_dim,
    data_path,
    epoch,
    model_path,
    signal_length,
    logger,
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=SEED, workers=True)

    model = ResNet(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c1_groups=c1_groups,
        c1_n_block=c1_n_block,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(
        val_check_interval=1.0,
        max_epochs=epoch,
        devices="auto",
        accelerator="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="training_loss", mode="min", check_on_train_epoch_end=True
            )
        ],
    )
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def load_model(model_name, model_path, gpu):
    if gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_name == 'mlp':
        base_class = MLP
    elif model_name == 'sae':
        base_class = SAE
    elif model_name == 'cnn':
        base_class = CNN
    elif model_name == 'resnet':
        base_class = ResNet

    model = (
        base_class.load_from_checkpoint(
            str(Path(model_path).absolute()), map_location=torch.device(device)
        )
        .float()
        .to(device)
    )

    model.eval()

    return model


def normalise_cm(cm):
    with np.errstate(all="ignore"):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm
