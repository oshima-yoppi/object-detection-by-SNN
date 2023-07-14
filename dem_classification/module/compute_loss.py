from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def spike_mse_loss(input, label, rate=0.8):
    # input shape:[time, batch, channel, pixel, pixel]?
    # print(label.shape)
    batch = label.shape[0]
    label = label.reshape(batch, -1)
    criterion = nn.MSELoss()
    num_steps = input.shape[0]
    input_spike_count = torch.sum(input, dim=0)
    target_spike_count = num_steps * rate * label
    target_spike_count = torch.round(target_spike_count)
    loss = criterion(input_spike_count, target_spike_count)
    return loss


def spike_count(spk_rec: torch.Tensor, channel=False):
    if channel == False:
        spk_rec = spk_rec.squeeze()
    count = torch.sum(spk_rec, dim=0,)
    return count


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
        f1 = (
            (1 + self.beta ** 2)
            * precision
            * recall
            / (self.beta ** 2 * precision + recall + self.smooth)
        )

        return 1 - f1


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


class AnalyzerClassification(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def get_accuracy(self, pred_pro, label):
        batch = len(label)
        pred_class = pred_pro.argmax(dim=1)
        label_class = label.argmax(dim=1)
        acc = torch.sum(pred_class == label_class) / batch
        acc = acc.item()
        return acc

    def get_precision(self, pred_pro, label):
        batch = len(label)
        pred_class = pred_pro.argmax(dim=1)
        label_class = label.argmax(dim=1)
        tp = torch.sum(pred_class * label_class)
        precision = (tp + self.smooth) / (torch.sum(pred_class) + self.smooth)
        precision = precision.item()
        return precision

    def get_recall(self, pred_pro, label):
        batch = len(label)
        pred_class = pred_pro.argmax(dim=1)
        label_class = label.argmax(dim=1)
        tp = torch.sum(pred_class * label_class)
        recall = (tp + self.smooth) / (torch.sum(label_class) + self.smooth)
        recall = recall.item()
        return recall

    def forward(self, pred_pro, label):
        acc = self.get_accuracy(pred_pro, label)
        precision = self.get_precision(pred_pro, label)
        recall = self.get_recall(pred_pro, label)
        return acc, precision, recall


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


class SegmentationAnalyzer:
    def __init__(self, binary_rate=0.5, smooth=1e-5):
        self.smooth = smooth
        self.binary_rate = binary_rate

    def _get_BinaryMap(self, pred_pro, label):
        self.pred_binary = torch.where(pred_pro[:, 1] > self.binary_rate, 1, 0)
        self.pred_binary = self.pred_binary.reshape(-1)
        self.label = label.reshape(-1)
        return

    def get_iou(self,):

        union = torch.logical_or(self.pred_binary, self.label).sum()
        intersection = torch.logical_and(self.pred_binary, self.label).sum()
        eps = 1e-6
        iou = torch.mean(intersection + eps / (union + eps))
        return iou.item()

    def get_precsion(self,):

        intersection = torch.logical_and(self.pred_binary, self.label).sum()
        eps = 1e-6
        prec = torch.mean(intersection + eps / (self.pred_binary.sum() + eps))
        return prec.item()

    def get_recall(self,):
        intersection = torch.logical_and(self.pred_binary, self.label).sum()
        eps = 1e-6
        recall = torch.mean((intersection + eps) / (self.label.sum() + eps))
        return recall.item()

    def __call__(self, pred_pro, label):
        self._get_BinaryMap(pred_pro, label)
        iou = self.get_iou()
        prec = self.get_precsion()
        recall = self.get_recall()
        return iou, prec, recall


# class ClassificationAnalyzer():
#     def __init__(self, binary_rate=0.5, smooth=1e-5):
#         self.smooth = smooth
#         self.binary_rate = binary_rate
#     def _get_BinaryMap(self, pred_pro, label):
#         self.pred_binary = torch.where(pred_pro[:,1]>self.binary_rate, 1, 0)
#         self.pred_binary = self.pred_binary.reshape(-1)
#         self.label = label.reshape(-1)
#         return
#     def get_iou(self, ):

#         union  = torch.logical_or(self.pred_binary, self.label).sum()
#         intersection = torch.logical_and(self.pred_binary, self.label).sum()
#         eps = 1e-6
#         iou = torch.mean(intersection+eps/(union+eps))
#         return iou.item()
#     def get_precsion(self,):

#         intersection = torch.logical_and(self.pred_binary, self.label).sum()
#         eps = 1e-6
#         prec = torch.mean(intersection+eps/(self.pred_binary.sum()+eps))
#         return prec.item()

#     def get_recall(self,):
#         intersection = torch.logical_and(self.pred_binary, self.label).sum()
#         eps = 1e-6
#         recall = torch.mean((intersection+eps)/(self.label.sum()+eps))
#         return recall.item()

#     def __call__(self, pred_pro, label):
#         self._get_BinaryMap(pred_pro, label)
#         iou = self.get_iou()
#         prec = self.get_precsion()
#         recall = self.get_recall()
#         return iou, prec,  recall


def culc_iou(pred_pro, label, rate=0.8):

    batch = len(label)
    # print(pred_pro.shape) #batch, channel , pixel, pixel
    pred_pro = pred_pro[:, 1, :, :]
    pred_pro = pred_pro.reshape(batch, -1)
    label = label.reshape(batch, -1)
    pred_binary = torch.where(pred_pro >= rate, 1, 0)
    union = torch.logical_or(pred_binary, label).sum(dim=1)
    intersection = torch.logical_and(pred_binary, label).sum(dim=1)
    eps = 1e-6
    iou = torch.mean(intersection / (union + eps))
    return iou.item()


if __name__ == "__main__":
    a = torch.ones((16, 2, 64, 64))
    label = torch.ones((16, 64, 64))
    print(culc_iou(a, label))
