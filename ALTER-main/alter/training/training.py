from alter.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from alter.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from alter.components import LRScheduler
import logging


class Train:
    """
    Training and evaluation pipeline for the model
    Supports training, validation, testing, metric recording, and result saving
    Compatible with 10-fold cross-validation
    """

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:
        """
        Initialize training configuration, model, dataloaders, and utilities
        Args:
            cfg: Hydra configuration object
            model: Neural network model
            optimizers: List of optimizers
            lr_schedulers: List of learning rate schedulers
            dataloaders: List containing train, val, test dataloaders
            logger: Logger for recording training logs
        """
        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')

        # Load train, validation, test dataloaders
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        # Get current fold index from config for 10-fold cross-validation
        self.fold = getattr(cfg, 'fold', 0)
        # Create independent save path for each fold
        self.save_path = Path(cfg.log_path) / cfg.unique_id / f"fold_{self.fold}"

        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()

    def init_meters(self):
        """Initialize metric meters for loss and accuracy recording"""
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    def reset_meters(self):
        """Reset all metric meters before each epoch"""
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        """
        Training process for one epoch
        Args:
            optimizer: Model optimizer
            lr_scheduler: Learning rate scheduler
        """
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            # Update learning rate
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            # Move data to GPU
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            # Apply mixup augmentation if enabled
            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            # Forward propagation
            predict = self.model(time_series, node_feature)

            # Calculate loss
            loss = self.loss_fn(predict, label)

            # Update training metrics
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        """
        Evaluation process for one epoch
        Args:
            dataloader: DataLoader for validation/test set
            loss_meter: Meter to record loss
            acc_meter: Meter to record accuracy
        Returns:
            List of evaluation metrics: AUC, precision, recall, F1, etc.
        """
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = self.model(time_series, node_feature)

            label = label.float()

            # Calculate loss
            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])

            # Collect predictions and labels for metric calculation
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        # Calculate AUC
        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0

        # Calculate precision, recall, F1-score
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        # Generate classification report
        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):
        """Save the learnable adjacency matrix from the model for analysis"""
        learable_matrixs = []
        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        """
        Save training process records and model weights
        Args:
            results: Training metrics throughout all epochs
        """
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def save_fold_final_result(self, final_metrics: dict):
        """
        Save and print final evaluation metrics for the current fold
        Args:
            final_metrics: Dictionary of final metrics for the fold
        """
        self.save_path.mkdir(exist_ok=True, parents=True)
        # Save fold result to TXT file
        result_file = self.save_path / "fold_final_result.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"===== Fold {self.fold} Final Result =====\n")
            for key, value in final_metrics.items():
                f.write(f"{key}: {value}\n")
        # Print fold result to console
        self.logger.info(f"\n===== Fold {self.fold} Final Evaluation Result =====")
        for key, value in final_metrics.items():
            self.logger.info(f"{key}: {value:.4f}")

    def train(self):
        """Main training loop: iterate over all epochs and complete training/validation/testing"""
        training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            # Train one epoch
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            # Validate one epoch
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)
            # Test one epoch
            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            # Print epoch logs
            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            # Log metrics to wandb
            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Val AUC": val_result[0],
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            # Record training process
            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        # Save final metrics of current fold
        final_result = {
            "Fold": self.fold,
            "Final Train Loss": self.train_loss.avg,
            "Final Train Accuracy": self.train_accuracy.avg,
            "Final Test Loss": self.test_loss.avg,
            "Final Test Accuracy": self.test_accuracy.avg,
            "Final Test AUC": test_result[0],
            "Final Test Sensitivity": test_result[-1],
            "Final Test Specificity": test_result[-2],
            "Final Micro Precision": test_result[-6],
            "Final Micro Recall": test_result[-5],
            "Final Micro F1": test_result[-4],
            "Final Val AUC": val_result[0]
        }
        self.save_fold_final_result(final_result)

        # Save learnable graph matrix if enabled
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)