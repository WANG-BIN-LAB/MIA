from datetime import datetime
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
import numpy as np
import time


def model_training(cfg: DictConfig, lds, ld1, ld2, fold: int):
    """
    Execute training process for a single fold in 10-fold cross-validation
    Args:
        cfg: Hydra configuration object
        lds: Hyperparameter for model training
        ld1: Hyperparameter for model training
        ld2: Hyperparameter for model training
        fold: Current fold index for cross-validation
    Returns:
        Evaluation metrics for current fold
    """
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S") + f"_fold_{fold}"

    # Get dataloaders for current fold
    dataloaders = dataset_factory(cfg, fold=fold)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)

    # Start training pipeline
    training_pipeline = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    fold_metrics = training_pipeline.train(lds, ld1, ld2, fold)
    return fold_metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 10-Fold Cross Validation Configuration
    NUM_FOLDS = 10
    # Container to store evaluation metrics of all folds
    metric_records = {
        "accuracy": [],
        "auc": [],
        "sensitivity": [],
        "specificity": [],
        "recall": [],
        "precision": [],
        "f1_macro": []
    }

    # Fixed hyperparameters for training
    ld1, ld2, lds = 1, 1, 0.000125

    print("\n" + "=" * 80)
    print("START 10-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 80)

    # Iterate over all 10 folds
    for fold_idx in range(NUM_FOLDS):
        print(f"\n>>> Start Training Fold {fold_idx + 1}/{NUM_FOLDS}")
        acc, auc, sen, spec, rec, prec, f1 = model_training(cfg, lds, ld1, ld2, fold_idx)

        # Save metrics of current fold
        metric_records["accuracy"].append(acc)
        metric_records["auc"].append(auc)
        metric_records["sensitivity"].append(sen)
        metric_records["specificity"].append(spec)
        metric_records["recall"].append(rec)
        metric_records["precision"].append(prec)
        metric_records["f1_macro"].append(f1)

    # Calculate and print final statistics: Mean ± Standard Deviation
    print("\n" + "=" * 80)
    print("FINAL 10-FOLD CROSS-VALIDATION RESULTS (MEAN ± STD)")
    print("=" * 80)
    print(f"Test Accuracy:      {np.mean(metric_records['accuracy']):.4f} ± {np.std(metric_records['accuracy']):.4f}")
    print(f"Test AUC:           {np.mean(metric_records['auc'])*100:.2f} ± {np.std(metric_records['auc'])*100:.2f} %")
    print(f"Sensitivity:        {np.mean(metric_records['sensitivity'])*100:.2f} ± {np.std(metric_records['sensitivity'])*100:.2f} %")
    print(f"Specificity:        {np.mean(metric_records['specificity'])*100:.2f} ± {np.std(metric_records['specificity'])*100:.2f} %")
    print(f"Recall:             {np.mean(metric_records['recall'])*100:.2f} ± {np.std(metric_records['recall'])*100:.2f} %")
    print(f"Precision:          {np.mean(metric_records['precision'])*100:.2f} ± {np.std(metric_records['precision'])*100:.2f} %")
    print(f"F1-Macro:           {np.mean(metric_records['f1_macro'])*100:.2f} ± {np.std(metric_records['f1_macro'])*100:.2f} %")
    print("=" * 80)


if __name__ == '__main__':
    total_start = time.time()
    main()
    total_end = time.time()
    total_sec = total_end - total_start
    total_hour = total_sec / 3600
    print(f"\nTotal Running Time: {total_sec:.2f} seconds | {total_hour:.2f} hours")