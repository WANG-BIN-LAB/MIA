from omegaconf import DictConfig, open_dict
from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils.data as utils
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

def dataset_factory(cfg: DictConfig, fold: int = None) -> List[utils.DataLoader]:
    """
    Dataset factory supporting 5-fold stratified cross-validation
    Args:
        cfg: Hydra configuration
        fold: Fold index (0-4) for cross-validation; None = original train/val/test split
    Returns:
        List of DataLoaders: [train_loader, val_loader, test_loader]
    """
    # Validate dataset name
    assert cfg.dataset.name in ['abide', 'hcp'], "Dataset must be either 'abide' or 'hcp'"

    # Load full dataset (original logic preserved)
    load_data_func = eval(f"load_{cfg.dataset.name}_data")
    final_timeseires, final_pearson, labels, stratified = load_data_func(cfg)

    # ====================== 5-Fold Stratified Cross-Validation Mode ======================
    if fold is not None:
        # One-hot encode labels
        onehot_labels = F.one_hot(labels.to(torch.int64))
        total_samples = len(final_timeseires)

        # Stratified 5-fold split (preserves class balance)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_splits = list(skf.split(np.zeros(total_samples), stratified))
        train_val_idx, test_idx = fold_splits[fold]

        # Split train/validation (90% train, 10% val from training set)
        val_size = int(len(train_val_idx) * 0.1)
        train_idx = train_val_idx[:-val_size]
        val_idx = train_val_idx[-val_size:]

        # Create fold-specific datasets
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
            val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False
        )
        test_loader = utils.DataLoader(
            test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False
        )

        # Update training steps in config
        with open_dict(cfg):
            cfg.steps_per_epoch = (len(train_idx) - 1) // cfg.dataset.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

        return [train_loader, val_loader, test_loader]

    # ====================== Original Train/Val/Test Split Mode ======================
    else:
        dataloaders = init_stratified_dataloader(cfg, final_timeseires, final_pearson, labels, stratified) \
            if cfg.dataset.stratified \
            else init_dataloader(cfg, final_timeseires, final_pearson, labels, stratified)
        return dataloaders