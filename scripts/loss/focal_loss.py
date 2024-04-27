from typing import Union, List

import torch
from torch import nn, Tensor
import numpy as np

'''
# Usage example:
focal_loss = FocalLossWithRobustness(gamma=2.0, alpha=0.25, ignore_index=-100)
loss = focal_loss(input_logits, target_tensor)
'''
class FocalLossWithRobustness(nn.Module):
    """
    This class implements the Focal Loss with the robustness adaptations from RobustCrossEntropyLoss.
    Input must be logits, not probabilities!

    Args:
        gamma (float): Focusing parameter. A higher value gives more weight to hard-to-classify examples. default:2.0 (when gamma=0, equals to cross entropy)
        alpha (float or list[float]): Weighting factor for positive examples. Can be a single value or a list of values per class.
        ignore_index (int): Index to ignore in the target tensor.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    '''
    def focal_loss(y_real, y_pred, eps=1e-8, gamma=2):
        y_pred = torch.clamp(y_pred, min=eps, max=1 - eps)
        y_pred = torch.sigmoid(y_pred)
        y_real = torch.clamp(y_real, min=eps)
        your_loss = -torch.sum(
            ((1 - y_pred) ** gamma) * y_real * torch.log(y_pred) + (y_pred ** gamma) * (1 - y_real) * torch.log(
                1 - y_pred))

        return your_loss
    '''

    def __init__(self, gamma: float = 2.0, alpha: Union[float, List[float]] = 0.25, ignore_index: int = -100, reduction: str = "mean"):
        super(FocalLossWithRobustness, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        target = target.long()

        # Compute probabilities from logits
        log_probs = torch.nn.functional.log_softmax(input, dim=1)

        # Mask out ignored indices
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        log_probs = log_probs[valid_mask]

        # Compute focal loss
        probs = torch.exp(log_probs)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_loss = -self.alpha * (1 - pt)**self.gamma * log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

