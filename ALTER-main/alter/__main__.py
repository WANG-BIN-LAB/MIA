# Import basic modules and third-party libraries
from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict

# Import custom factory functions
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory
from .components import logger_factory
from .training import training_factory

import numpy as np


def model_training(cfg: DictConfig):
    """
    Training pipeline for a single fold in 10-fold cross-validation
    Args:
        cfg: Global config with current fold index
    """
    # Generate a unique ID for the experiment run
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
        cfg.n_folds = 10  # Set to 10 folds

    # Automatically load train/val/test data for CURRENT fold (based on cfg.fold)
    dataloaders = dataset_factory(cfg)

    # Initialize training components
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)

    # Start training
    training = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Project entry:
    - Multiple repeated experiments
    - 10-fold stratified cross-validation (handled by dataset loader)
    """
    # WandB group name for result classification
    group_name = (
        f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    )

    # Loop: repeated experiments
    for _ in range(cfg.repeat_time):
        # Loop: 10-fold cross-validation (fold 0 ~ 9)
        for fold_idx in range(10):  # 10 folds
            # Pass current fold index to config → dataset uses this to split data
            with open_dict(cfg):
                cfg.fold = fold_idx

            # Initialize WandB log
            run = wandb.init(
                project=cfg.project,
                reinit=True,
                group=group_name,
                name=f"fold_{fold_idx}",
                tags=[cfg.dataset.name, "10-fold-CV"]  # 10-fold-CV
            )

            # Train on the current fold
            model_training(cfg)

            # Close WandB run
            run.finish()


if __name__ == '__main__':
    main()