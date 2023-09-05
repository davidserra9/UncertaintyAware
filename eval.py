import os
import torch
import time
import sys
import argparse
from datetime import timedelta
from hydra import compose, initialize
from torch import nn
from omegaconf import OmegaConf
from src.logging import logger
from src.models import get_model
from src.ICM_dataset import ICMDataset
from torch.utils.data import DataLoader
from src.training import valid_epoch, eval_uncertainty_model

def main(cfg) -> None:

    # Find which device is used
    if torch.cuda.is_available() and cfg.base.device == "cuda":
        logger.info(f'Evaluating the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    # Create the model
    model = get_model(cfg.model.encoder)
    model = model.to(cfg.base.device)

    # Load loss, optimizer and scheduler
    criterion = getattr(nn, cfg.training.loss)()

    # Load evaluation dataset
    eval_dataset = ICMDataset(path=os.path.join(cfg.base.dataset, "valid"),
                              train=False,
                              species=cfg.base.classes)

    if cfg.uncertainty.eval_dataloader.batch_size != 1:
        logger.warn("The test batch size must be 1. Changing it to 1.")
        cfg.uncertainty.eval_dataloader.batch_size = 1

    eval_loader = torch.utils.data.DataLoader(eval_dataset, **cfg.uncertainty.eval_dataloader)

    logger.info("===== Starting evaluation =====")
    start = time.time()
    acc, f1, loss, cm = valid_epoch(model,
                                    eval_loader,
                                    criterion,
                                    log_step=cfg.training.log_step,
                                    epoch=0,
                                    wb_log=False,
                                    cls_names=cfg.base.classes,
                                    device=cfg.base.device)

    cm.savefig(os.path.join(logger.output_path, "test_confusion_matrix.jpg"))

    logger.info(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Loss: {loss:.4f}")

    eval_uncertainty_model(model=model,
                           eval_loader=eval_loader,
                           mc_samples=cfg.uncertainty.mc_samples,
                           dropout_rate=cfg.uncertainty.dropout_rate,
                           num_classes=len(cfg.base.classes),
                           wb_log=False,
                           output_path=logger.output_path,
                           device=cfg.base.device)

    logger.info(f"===== Evaluation finished in {timedelta(seconds=round(time.time() - start))} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model following the instructions in the README file")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config

    initialize(version_base=None, config_path="config", job_name="training")
    config = compose(config_name=config_name)
    config = OmegaConf.create(config)
    main(config)

