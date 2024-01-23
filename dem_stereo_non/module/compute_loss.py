from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# def spike_mse_loss(input, target, rate=0.8):
#     # input shape:[time, batch, channel, pixel, pixel]?
#     # print(target.shape)
#     batch = target.shape[0]
#     target = target.reshape(batch, -1)
#     criterion = nn.MSELoss()
#     num_steps = input.shape[0]
#     input_spike_count = torch.sum(input, dim=0)
#     target_spike_count = num_steps * rate * target
#     target_spike_count = torch.round(target_spike_count)
#     loss = criterion(input_spike_count, target_spike_count)
#     return loss


def spike_count(spk_rec: torch.Tensor, channel=False):
    if channel == False:
        spk_rec = spk_rec.squeeze()
    count = torch.sum(
        spk_rec,
        dim=0,
    )
    return count


class BCELoss_Recall(nn.Module):
    def __init__(self, recall_rate):
        super().__init__()
        self.bce = nn.BCELoss()
        self.recall_rate = recall_rate
        self.smooth = 1e-5

    def _recall_loss(self, pred_pro, label):
        true_positives = torch.sum(label * pred_pro)
        actual_positives = torch.sum(label)
        recall = (true_positives + self.smooth) / (
            actual_positives + self.smooth
        )  # 0で除算を避けるため、小さな値を足しています
        loss = 1.0 - recall
        return loss

    def forward(self, pred_pro, label):
        loss = self.bce(pred_pro, label) + self.recall_rate * self._recall_loss(
            pred_pro, label
        )
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_pro, label):
        pred_pro = pred_pro.view(-1)
        label = label.view(-1)
        intersection = (pred_pro * label).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_pro.sum() + label.sum() + self.smooth
        )
        dice = 1 - dice
        return dice


class WeightedF1Loss(nn.Module):
    def __init__(self, beta, smooth=1e-5):
        """
        precision と recall の調和平均を考慮した損失関数
        beta：重みを付ける
        0< beta < 1 : precisionを重視
        1 < beta : recallを重視
        """
        super().__init__()
        self.smooth = smooth
        self.beta = beta

    def forward(self, pred_pro, label):
        # https://data.gunosy.io/entry/2016/08/05/115345
        # https://gucci-j.github.io/imbalanced-analysis/
        # https://atmarkit.itmedia.co.jp/ait/articles/2211/02/news027.html
        tp = torch.sum(pred_pro * label)
        fp = torch.sum(pred_pro) - tp
        fn = torch.sum(label) - tp
        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)
        # print(precision, recall, tp, fp, fn)
        # print(precision, recall)
        # print()
        # print(precision.item(), recall.item())
        f1 = ((1 + self.beta**2) * precision * recall + self.smooth) / (
            self.beta**2 * precision + recall + self.smooth
        )

        return 1 - f1


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(
            inputs, targets, reduction="mean", weight=self.weight
        )
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class Analyzer:
    def __init__(self, binary_rate=0.5, smooth=1e-5):
        self.smooth = smooth
        self.binary_rate = binary_rate

    def _get_BinaryMap(self, pred_pro, target):
        self.pred_binary = torch.where(pred_pro > self.binary_rate, 1, 0)
        self.pred_binary = self.pred_binary.reshape(-1)
        self.target = target.reshape(-1)
        # print(self.pred_binary.shape, self.target.shape)
        return

    def get_iou(
        self,
    ):
        union = torch.logical_or(self.pred_binary, self.target).sum()
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        iou = torch.mean((intersection + eps) / (union + eps))
        return iou.item()

    def get_precsion(
        self,
    ):
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        prec = torch.mean((intersection + eps) / (self.pred_binary.sum() + eps))
        return prec.item()

    def get_recall(
        self,
    ):
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        recall = torch.mean((intersection + eps) / (self.target.sum() + eps))
        return recall.item()

    def __call__(self, pred_pro, target):
        self._get_BinaryMap(pred_pro, target)
        iou = self.get_iou()
        prec = self.get_precsion()
        recall = self.get_recall()
        return iou, prec, recall
