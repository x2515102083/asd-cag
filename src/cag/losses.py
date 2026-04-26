from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.autograd import Function


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long().view(-1)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce)
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


def gradient_reverse(input_tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
    return GradientReversalFunction.apply(input_tensor, float(lambda_value))


def accuracy(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    return float(accuracy_score(labels.astype(int), (probabilities >= threshold).astype(int)))


def auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels.astype(int)).size < 2:
        return float("nan")
    return float(roc_auc_score(labels.astype(int), probabilities.astype(float)))


def specificity(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    tn, fp, _, _ = confusion_matrix(labels.astype(int), (probabilities >= threshold).astype(int), labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def sensitivity(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    _, _, fn, tp = confusion_matrix(labels.astype(int), (probabilities >= threshold).astype(int), labels=[0, 1]).ravel()
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return {
        "ACC": accuracy(labels, probabilities, threshold),
        "AUC": auc(labels, probabilities),
        "SPE": specificity(labels, probabilities, threshold),
        "SEN": sensitivity(labels, probabilities, threshold),
    }
