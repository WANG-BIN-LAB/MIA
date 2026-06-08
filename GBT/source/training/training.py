from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from source.training.LRR import LRRLoss
from sklearn import metrics


class Train:
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = dataloaders
        self.total_epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_dir = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_matrix = cfg.save_learnable_graph
        self.init_meters()

    def init_meters(self):
        """Initialize loss and accuracy meters"""
        self.train_loss, self.val_loss, self.test_loss, \
        self.train_acc, self.val_acc, self.test_acc = [TotalMeter() for _ in range(6)]

    def reset_meters(self):
        """Reset all meters before each epoch"""
        for meter in [self.train_loss, self.val_loss, self.test_loss,
                      self.train_acc, self.val_acc, self.test_acc]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler, lds, ld1, ld2):
        """Train one single epoch"""
        self.model.train()
        for time_series, node_feature, label in self.train_loader:
            label = label.float()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            # Move data to GPU
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            # Continuous mixup augmentation
            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(time_series, node_feature, y=label)

            # Compute LRR loss
            feature_flat = node_feature.view(node_feature.size(0), -1)
            target_label = np.argmax(label.cpu().numpy(), axis=1)
            lrr_loss = torch.squeeze(LRRLoss().apply(feature_flat, target_label, lds))

            # Model forward & total loss
            pred = self.model(time_series, node_feature)
            total_loss = self.loss_function(pred, label) + lrr_loss

            # Backward propagation and optimize
            self.train_loss.update_with_weight(total_loss.item(), label.shape[0])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Calculate training accuracy
            top1_acc = accuracy(pred, label[:, 1])[0]
            self.train_acc.update_with_weight(top1_acc, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter, lds, ld1, ld2):
        """Validation / Test inference for one epoch"""
        gt_labels = []
        pred_scores = []
        self.model.eval()

        with torch.no_grad():
            for time_series, node_feature, label in dataloader:
                time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.float().cuda()

                output = self.model(time_series, node_feature)
                # Compute LRR loss
                feature_flat = node_feature.view(node_feature.size(0), -1)
                target_label = np.argmax(label.cpu().numpy(), axis=1)
                lrr_loss = torch.squeeze(LRRLoss().apply(feature_flat, target_label, lds))
                # Total loss
                total_loss = self.loss_function(output, label) + lrr_loss

                loss_meter.update_with_weight(total_loss.item(), label.shape[0])
                top1_acc = accuracy(output, label[:, 1])[0]
                acc_meter.update_with_weight(top1_acc, label.shape[0])

                # Collect prediction scores and ground truth
                pred_scores += F.softmax(output, dim=1)[:, 1].tolist()
                gt_labels += label[:, 1].tolist()

        # Compute evaluation metrics
        auc_score = roc_auc_score(gt_labels, pred_scores)
        pred_binary = (np.array(pred_scores) > 0.5).astype(int)
        gt_binary = np.array(gt_labels)

        micro_metrics = precision_recall_fscore_support(gt_binary, pred_binary, average='micro')
        f1_macro = metrics.f1_score(gt_binary, pred_binary, average='macro')

        # Get sensitivity and specificity from classification report
        cls_report = classification_report(gt_binary, pred_binary, output_dict=True, zero_division=0)
        class_recall = [0.0, 0.0]
        for key in cls_report:
            if isfloat(key):
                class_recall[int(float(key))] = cls_report[key]['recall']

        return [auc_score] + list(micro_metrics) + class_recall, f1_macro

    def generate_save_learnable_matrix(self):
        """Save learnable matrix to local file"""
        matrix_list = []
        label_list = []
        for time_series, node_feature, label in self.test_loader:
            time_series, node_feature = time_series.cuda(), node_feature.cuda()
            _, mat, _ = self.model(time_series, node_feature)
            matrix_list.append(mat.cpu().detach().numpy())
            label_list += label.tolist()

        self.save_dir.mkdir(exist_ok=True, parents=True)
        np.save(
            self.save_dir / "learnable_matrix.npy",
            {"matrix": np.vstack(matrix_list), "label": np.array(label_list)},
            allow_pickle=True
        )

    def save_training_results(self, results):
        """Save training log and model checkpoint"""
        self.save_dir.mkdir(exist_ok=True, parents=True)
        np.save(self.save_dir / "training_process.npy", results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_dir / "model.pt")

    def train(self, lds, ld1, ld2, fold: int):
        """Main training pipeline for single fold"""
        training_log = []
        self.current_step = 0

        # Initialize best metrics
        best_val_auc = 0.0
        best_test_metrics = [0.0] * 7  # acc, auc, sen, spec, recall, precision, f1_macro

        print(f"\n--- Fold {fold + 1} Training Started ---")

        for epoch in range(self.total_epochs):
            self.reset_meters()
            # Train phase
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0], lds, ld1, ld2)
            # Validation phase
            val_results, val_f1 = self.test_per_epoch(self.val_loader, self.val_loss, self.val_acc, lds, ld1, ld2)
            # Test phase
            test_results, test_f1 = self.test_per_epoch(self.test_loader, self.test_loss, self.test_acc, lds, ld1, ld2)

            # Print epoch log
            if epoch % 1 == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{self.total_epochs}] | "
                    f"Train Loss: {self.train_loss.avg:.3f} | "
                    f"Train Acc: {self.train_acc.avg:.3f}% | "
                    f"Test Loss: {self.test_loss.avg:.3f} | "
                    f"Test Acc: {self.test_acc.avg:.3f}% | "
                    f"Test AUC: {test_results[0]:.4f} | "
                    f"Test Sensitivity: {test_results[-1]:.4f}"
                )

            # Update best model (monitor validation AUC)
            if val_results[0] > best_val_auc:
                best_val_auc = val_results[0]
                best_test_metrics = [
                    self.test_acc.avg,
                    test_results[0],
                    test_results[-1],
                    test_results[-2],
                    test_results[-5],
                    test_results[-6],
                    test_f1
                ]
                print(f"[Fold {fold + 1}] New Best Test Accuracy: {best_test_metrics[0]:.4f}")

            # Record training process
            training_log.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_acc.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_acc.avg,
                "Test AUC": test_results[0],
                "Test Sensitivity": test_results[-1],
                "Test Specificity": test_results[-2],
                "Test Recall": test_results[-5],
                "Test Precision": test_results[-6],
                "Test F1-Macro": test_f1,
                "Validation AUC": val_results[0]
            })

        # Print FULL metrics for current fold
        print(f"\n>>> Fold {fold + 1} Best Test Metrics Summary")
        print(f"Accuracy:    {best_test_metrics[0]:.4f}")
        print(f"AUC:         {best_test_metrics[1]*100:.2f} %")
        print(f"Sensitivity: {best_test_metrics[2]*100:.2f} %")
        print(f"Specificity: {best_test_metrics[3]*100:.2f} %")
        print(f"Recall:      {best_test_metrics[4]*100:.2f} %")
        print(f"Precision:   {best_test_metrics[5]*100:.2f} %")
        print(f"F1-Macro:    {best_test_metrics[6]*100:.2f} %")
        print("-" * 60)

        # Save files
        if self.save_learnable_matrix:
            self.generate_save_learnable_matrix()
        self.save_training_results(training_log)

        return best_test_metrics