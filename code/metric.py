import torch.nn as nn


class DiceCoeff(nn.Module):
    """
    The ordinary dice coefficients.

    Notice that the y_pred in the evulation phase is different from the training phase:
    during inference, for sigmoid activation, we can simply threshold the value by 0.5;
    for softmax activation, we must take the argmax along the channel dimension.
    """
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        
        return (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)


class WeightedDiceCoeff(nn.Module):
    """
    This is the wighted dice coefficients for multi-class segmentation, 
    which mitigate the problem of imbalance classes.
    @Param dim: The spatial dimension 
    @Param smooth: the smoothness added to the dice score

    Notice that the y_pred should be the value after the activation function, 
    either sigmoid(single class) or softmax(multi-class).
    """
    def __init__(self, dim=(-2, -1), smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.dim = dim

    def forward(self, y_true, y_pred):
        intersection = 2. * (y_true * y_pred).sum(dim=self.dim) + self.smooth / 2.
        union = y_true.sum(dim=self.dim) + y_pred.sum(dim=self.dim) + self.smooth

        return (intersection / union).mean()


class WeightedDiceLoss(nn.Module):
    """
    The weighted dice loss which is 1 - dice_coefficients
    """
    def __init__(self, dim=(-2, -1), smooth=1e-6):
        super().__init__()
        self.weighted_dice_coeff = WeightedDiceCoeff(dim, smooth)

    def forward(self, y_true, y_pred):
        return 1 - self.weighted_dice_coeff(y_true, y_pred)