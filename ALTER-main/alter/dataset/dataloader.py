import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.nn.functional as F


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    """
    Create dataloaders using 10-fold stratified cross-validation
    Args:
        cfg: Hydra config
        final_timeseires: Time series features
        final_pearson: Pearson matrix features
        labels: Ground truth labels
    Returns: train_loader, val_loader, val_loader (as test)
    """
    # One-hot encoding for labels
    labels = F.one_hot(labels.to(torch.int64))
    labels_np = labels.argmax(dim=1).cpu().numpy()

    # Fixed 10-fold split with reproducible seed
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds
    folds = list(skf.split(final_timeseires, labels_np))
    train_idx, val_idx = folds[cfg.fold]

    # Calculate scheduler parameters
    with open_dict(cfg):
        cfg.steps_per_epoch = (len(train_idx) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # Full dataset
    dataset = utils.TensorDataset(final_timeseires, final_pearson, labels)

    # Subset for current fold
    train_dataset = utils.Subset(dataset, train_idx)
    val_dataset = utils.Subset(dataset, val_idx)

    # Build dataloaders
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    # Keep return format consistent: [train, val, test]
    return [train_dataloader, val_dataloader, val_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    """
    Stratified dataloader for balanced label distribution in 10-fold CV
    Args:
        cfg: Hydra config
        final_timeseires: Time series features
        final_pearson: Pearson matrix features
        labels: Ground truth labels
        stratified: Stratified array for balanced splitting
    Returns: train_loader, val_loader, val_loader (as test)
    """
    # One-hot encoding for labels
    labels = F.one_hot(labels.to(torch.int64))

    # Fixed 10-fold split with reproducible seed
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds
    folds = list(skf.split(final_timeseires, stratified))
    train_idx, val_idx = folds[cfg.fold]

    # Calculate scheduler parameters
    with open_dict(cfg):
        cfg.steps_per_epoch = (len(train_idx) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # Full dataset
    dataset = utils.TensorDataset(final_timeseires, final_pearson, labels)

    # Subset for current fold
    train_dataset = utils.Subset(dataset, train_idx)
    val_dataset = utils.Subset(dataset, val_idx)

    # Build dataloaders
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    # Keep return format consistent: [train, val, test]
    return [train_dataloader, val_dataloader, val_dataloader]