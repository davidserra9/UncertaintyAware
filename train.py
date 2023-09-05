import os
import sys
import torch
import wandb
import argparse

from torch import nn
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.logging import logger
from src.models import get_model
from src.training import get_optimizer, get_scheduler, fit, eval_uncertainty_model
from src.ICM_dataset import ICMDataset

def main(cfg) -> None:

    # Check which device is used
    if torch.cuda.is_available() and "cuda" in cfg.base.device:
        logger.info(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    if "wandb" in cfg.base:

        num_exp = len(wandb.Api().runs(cfg.base.wandb.params.project))
        logger.info(f"Starting experiment {cfg.model.encoder.name}_{num_exp:02} on WANDB.")
        logger.info(f"Project: {cfg.base.wandb.params.project}. Entity: {cfg.base.wandb.params.entity}")
        wandb.init(project=cfg.base.wandb.params.project,
                   entity=cfg.base.wandb.params.entity,
                   name=f"{cfg.model.encoder.name}_{num_exp:02}",
                   config=OmegaConf.to_container(cfg.training))
        wandb_run_name = wandb.run.name

    # Create the model
    model = get_model(cfg.model.encoder)
    model = model.to(cfg.base.device)

    # Load loss, optimizer and scheduler
    criterion = getattr(nn, cfg.training.loss)()
    optimizer = get_optimizer(model, cfg.training.optimizer)
    scheduler = get_scheduler(optimizer, cfg.training.scheduler)

    # Load datasets
    train_dataset = ICMDataset(path=os.path.join(cfg.base.dataset, "train"),
                               train=True,
                               oversample=cfg.training.oversample,
                               species=cfg.base.classes)

    valid_dataset = ICMDataset(path=os.path.join(cfg.base.dataset, "valid"),
                               train=False,
                               species=cfg.base.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, **cfg.training.train_dataloader)

    if cfg.training.valid_dataloader.batch_size != 1:
        logger.warning("The valid batch size must be 1. Changing it to 1.")
        cfg.training.valid_dataloader.batch_size = 1

    valid_loader = torch.utils.data.DataLoader(valid_dataset, **cfg.training.valid_dataloader)

    # Train the model
    fit(model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.training.epochs,
        wb_log="wandb" in cfg.base,
        log_step=cfg.training.log_step,
        cls_names=cfg.base.classes,
        output_path=logger.output_path,
        device=cfg.base.device)

    eval_dataset = ICMDataset(path=os.path.join(cfg.base.dataset, "test"),
                              train=False,
                              species=cfg.base.classes)

    if cfg.uncertainty.eval_dataloader.batch_size != 1:
        logger.warn("The test batch size must be 1. Changing it to 1.")
        cfg.uncertainty.eval_dataloader.batch_size = 1

    eval_loader = torch.utils.data.DataLoader(eval_dataset, **cfg.uncertainty.eval_dataloader)

    eval_uncertainty_model(model=model,
                           eval_loader=eval_loader,
                           mc_samples=cfg.uncertainty.mc_samples,
                           dropout_rate=cfg.uncertainty.dropout_rate,
                           num_classes=len(cfg.base.classes),
                           wb_log="wandb" in cfg.base,
                           output_path=logger.output_path,
                           device=cfg.base.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model following the instructions in the README file")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config

    initialize(version_base=None, config_path="config", job_name="training")
    config = compose(config_name=config_name)
    config = OmegaConf.create(config)
    main(config)


