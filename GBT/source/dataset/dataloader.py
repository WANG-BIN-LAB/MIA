import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor,
                    stratified: np.array) -> List[utils.DataLoader]:
    """
    Initialize standard train/val/test dataloader with random split
    """
    labels = F.one_hot(labels.to(torch.int64))
    total_samples = final_timeseires.shape[0]
    train_size = int(total_samples * cfg.dataset.train_set * cfg.datasz.percentage)
    val_size = int(total_samples * cfg.dataset.val_set)

    if cfg.datasz.percentage == 1.0:
        test_size = total_samples - train_size - val_size
    else:
        test_size = int(total_samples * (1 - cfg.dataset.val_set - cfg.dataset.train_set))

    # Update training steps in config
    with open_dict(cfg):
        cfg.steps_per_epoch = (train_size - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # Build full dataset
    full_dataset = utils.TensorDataset(
        final_timeseires[:train_size + val_size + test_size],
        final_pearson[:train_size + val_size + test_size],
        labels[:train_size + val_size + test_size]
    )

    # Random split
    train_dataset, val_dataset, test_dataset = utils.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last
    )
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    """
    Initialize dataloader with stratified shuffle split to keep class balance
    """
    labels = F.one_hot(labels.to(torch.int64))
    total_samples = final_timeseires.shape[0]
    train_size = int(total_samples * cfg.dataset.train_set * cfg.datasz.percentage)
    val_size = int(total_samples * cfg.dataset.val_set)

    if cfg.datasz.percentage == 1.0:
        test_size = total_samples - train_size - val_size
    else:
        test_size = int(total_samples * (1 - cfg.dataset.val_set - cfg.dataset.train_set))

    # Update training steps in config
    with open_dict(cfg):
        cfg.steps_per_epoch = (train_size - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # First split: train | val+test
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size + test_size, train_size=train_size, random_state=42
    )
    for train_idx, val_test_idx in split.split(final_timeseires, stratified):
        ts_train, pearson_train, label_train = final_timeseires[train_idx], final_pearson[train_idx], labels[train_idx]
        ts_valtest, pearson_valtest, label_valtest = final_timeseires[val_test_idx], final_pearson[val_test_idx], labels[val_test_idx]
        stratified_valtest = stratified[val_test_idx]

    # Second split: val | test
    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )
    for test_idx, val_idx in split2.split(ts_valtest, stratified_valtest):
        ts_test, pearson_test, label_test = ts_valtest[test_idx], pearson_valtest[test_idx], label_valtest[test_idx]
        ts_val, pearson_val, label_val = ts_valtest[val_idx], pearson_valtest[val_idx], label_valtest[val_idx]

    # Build datasets
    train_dataset = utils.TensorDataset(ts_train, pearson_train, label_train)
    val_dataset = utils.TensorDataset(ts_val, pearson_val, label_val)
    test_dataset = utils.TensorDataset(ts_test, pearson_test, label_test)

    # Build dataloaders
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last
    )
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )

    return [train_dataloader, val_dataloader, test_dataloader]


def dataset_factory(cfg: DictConfig, fold: int = None) -> List[utils.DataLoader]:
    """
    Unified dataset factory:
        - fold=None: use original stratified shuffle split
        - fold=0~4: enable 5-fold stratified cross validation
    """
    # Load your original full dataset
    from ..data import get_dataset
    final_timeseires, final_pearson, labels, stratified = get_dataset(cfg)

    # Mode 1: 5-Fold Stratified Cross Validation
    if fold is not None:
        onehot_labels = F.one_hot(labels.to(torch.int64))
        total_samples = len(final_timeseires)

        # 5-fold split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_splits = list(skf.split(np.zeros(total_samples), stratified))
        train_val_idx, test_idx = fold_splits[fold]

        # Split train+val into train (90%) and val (10%)
        val_split_size = int(len(train_val_idx) * 0.1)
        train_idx = train_val_idx[:-val_split_size]
        val_idx = train_val_idx[-val_split_size:]

        # Create datasets for current fold
        train_dataset = utils.TensorDataset(
            final_timeseires[train_idx], final_pearson[train_idx], onehot_labels[train_idx]
        )
        val_dataset = utils.TensorDataset(
            final_timeseires[val_idx], final_pearson[val_idx], onehot_labels[val_idx]
        )
        test_dataset = utils.TensorDataset(
            final_timeseires[test_idx], final_pearson[test_idx], onehot_labels[test_idx]
        )

        # Create dataloaders
        train_loader = utils.DataLoader(
            train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last
        )
        val_loader = utils.DataLoader(
            val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False
        )
        test_loader = utils.DataLoader(
            test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False
        )

        # Update step count for training
        with open_dict(cfg):
            cfg.steps_per_epoch = (len(train_idx) - 1) // cfg.dataset.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

        return [train_loader, val_loader, test_loader]

    # Mode 2: Original stratified split
    else:
        return init_stratified_dataloader(cfg, final_timeseires, final_pearson, labels, stratified)