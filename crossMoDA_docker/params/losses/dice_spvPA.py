# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction, Weight


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        hardness_weight=None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.hardness_weight = hardness_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, dim=1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))

        if self.hardness_weight is not None:
            intersection = torch.sum(self.hardness_weight * target * input, dim=reduce_axis)
        else:
            intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        if self.hardness_weight is not None:
            ground_o = torch.sum(self.hardness_weight * target, dim=reduce_axis)
            pred_o = torch.sum(self.hardness_weight * input, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class Dice_spvPA(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        supervised_attention=True,
        hardness_weighting=True,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.supervised_attention = supervised_attention
        self.hardness_weighting = hardness_weighting

    def forward(self, input, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """

        x, att_maps = input

        loss_function_single_channel = Dice(to_onehot_y=False, softmax=False)

        total_att_loss = 0

        if self.supervised_attention:
            L = len(att_maps)
            att_losses = []

            G_l = target
            for level in range(L):
                # A[level] are the attention maps as they arrive here
                # G[level] are downsampled ground truth maps, they are converted to one-hot inside the loss-function
                att_loss = loss_function_single_channel(att_maps[L - level - 1], G_l)
                att_losses.append(att_loss)
                total_att_loss = total_att_loss + 1 / L * att_loss

                if level < L - 1:
                    shape_curr_att_map = att_maps[L - level - 1].shape
                    shape_next_att_map = att_maps[L - level - 2].shape

                    # assert that shape of current attention map is multiple of next att map in all dimensions
                    assert all([x % y == 0 for x, y in zip(shape_curr_att_map, shape_next_att_map)])
                    shape_ratio = [x // y for x, y in zip(shape_curr_att_map, shape_next_att_map)]

                    kernel_size_and_stride = shape_ratio[2:5]
                    G_l = torch.nn.MaxPool3d(kernel_size=kernel_size_and_stride, stride=kernel_size_and_stride)(G_l)
        hardness_weight = None
        if self.hardness_weighting:
            hardness_lambda = 0.6
            hardness_weight = hardness_lambda * abs(
                torch.softmax(x, dim=1) - one_hot(target, num_classes=x.shape[1])
            ) + (1.0 - hardness_lambda)
            # img = hardness_weight.cpu().detach().numpy()
            # x_ = torch.softmax(x, dim=1).cpu().detach().numpy()
            # target_ = one_hot(target, num_classes=x.shape[1]).cpu().detach().numpy()
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(x_[1, 1, :, :, 20], cmap='gray')
            # axs[1].imshow(target_[1, 1, :, :, 20])
            # axs[2].imshow(img[1, 1, :, :, 20])
            # plt.show()
            # pass

        loss_function_multi_channel = Dice(to_onehot_y=True, softmax=True, hardness_weight=hardness_weight)
        pred_loss = loss_function_multi_channel(x, target)
        return total_att_loss + pred_loss


class MaskedDiceLoss(DiceLoss):
    """
    Same as DiceLoss, but accepts a binary mask ([0,1]) indicating a region over which to compute the dice.
    """

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5, mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.
            mask: the shape should B1H[WD] or 11H[WD].
        """
        if mask is not None:
            # checking if mask is of proper shape
            assert input.dim() == mask.dim(), f"dim of input ({input.shape}) is different from mask ({mask.shape})"
            assert (
                input.shape[0] == mask.shape[0] or mask.shape[0] == 1
            ), f" batch size of mask ({mask.shape}) must be 1 or equal to input ({input.shape})"

            if target.dim() > 1:
                assert mask.shape[1] == 1, f"mask ({mask.shape}) must have only 1 channel"
                assert (
                    input.shape[2:] == mask.shape[2:]
                ), f"spatial size of input ({input.shape}) is different from mask ({mask.shape})"

            input = input * mask
            target = target * mask

        return super().forward(input=input, target=target, smooth=smooth)


class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        w_type: Union[Weight, str] = Weight.SQUARE,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        w_type = Weight(w_type)
        self.w_func: Callable = torch.ones_like
        if w_type == Weight.SIMPLE:
            self.w_func = torch.reciprocal
        elif w_type == Weight.SQUARE:
            self.w_func = lambda x: torch.reciprocal(x * x)

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f = 1.0 - (2.0 * (intersection * w).sum(1) + smooth) / ((denominator * w).sum(1) + smooth)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Generalized Wasserstein Dice Loss [1] in PyTorch.
    Compared to [1] we used a weighting method similar to the one
    used in the generalized Dice Loss [2].

    References:
    ===========
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017.
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations",
        Sudre C., et al. MICCAI DLMIA 2017.

    wasserstein_distance_map:
    Compute the voxel-wise Wasserstein distance (eq. 6 in [1]) between the
    flattened prediction and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.
    References:
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017

    compute_weights_generalized_true_positives:
    Compute the weights \alpha_l of eq. 9 in [1] but using the weighting
    method proposed in the generalized Dice Loss [2].
    References:
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations." Sudre C., et al.
        MICCAI DLMIA 2017.
    """

    def __init__(self, dist_matrix, reduction: Union[LossReduction, str] = LossReduction.MEAN):
        """
        Args:
            dist_matrix: 2d tensor or 2d numpy array; matrix of distances
                between the classes. It must have dimension C x C where C is the
                number of classes.
            reduction: str; reduction mode.

        Raises:
            ValueError: When ``dist_matrix`` is not a square matrix.

        """
        super(GeneralizedWassersteinDiceLoss, self).__init__(reduction=LossReduction(reduction).value)

        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError(f"dist_matrix must be C x C, got {dist_matrix.shape[0]} x {dist_matrix.shape[1]}.")

        self.m = dist_matrix
        if isinstance(self.m, np.ndarray):
            self.m = torch.from_numpy(self.m)
        if torch.max(self.m) != 1:
            self.m = self.m / torch.max(self.m)
        self.num_classes = self.m.size(0)

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        """
        # Aggregate spatial dimensions
        flat_input = input.view(input.size(0), input.size(1), -1)
        flat_target = target.view(target.size(0), -1)

        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)

        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)

        # Compute the generalised number of true positives
        alpha = self.compute_weights_generalized_true_positives(flat_target)
        true_pos = self.compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
        denom = self.compute_denominator(alpha, flat_target, wass_dist_map)

        # Compute and return the final loss
        wass_dice = (2.0 * true_pos + smooth) / (denom + smooth)
        wass_dice_loss = 1.0 - wass_dice
        return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba: torch.Tensor, flat_target: torch.Tensor):
        """
        Args:
            flat_proba: the probabilities of input(predicted) tensor.
            flat_target: the target tensor.
        """
        # Turn the distance matrix to a map of identical matrix
        m = torch.clone(self.m).to(flat_proba.device)
        m_extended = torch.unsqueeze(m, dim=0)
        m_extended = torch.unsqueeze(m_extended, dim=3)
        m_extended = m_extended.expand((flat_proba.size(0), m_extended.size(1), m_extended.size(2), flat_proba.size(2)))

        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        flat_target_extended = flat_target_extended.expand(
            (flat_target.size(0), m_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)

        # Extract the vector of class distances for the ground-truth label at each voxel
        m_extended = torch.gather(m_extended, dim=1, index=flat_target_extended)
        m_extended = torch.squeeze(m_extended, dim=1)

        # Compute the wasserstein distance map
        wasserstein_map = m_extended * flat_proba

        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)
        return wasserstein_map

    def compute_generalized_true_positive(
        self, alpha: torch.Tensor, flat_target: torch.Tensor, wasserstein_distance_map
    ):
        """
        Args:
            alpha: generalised number of true positives of target class.
            flat_target: the target tensor.
            wasserstein_distance_map: the map obtained from the above function.
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)
        alpha_extended = alpha_extended.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_extended, index=flat_target_extended, dim=1)

        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (1.0 - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_denominator(self, alpha: torch.Tensor, flat_target: torch.Tensor, wasserstein_distance_map):
        """
        Args:
            alpha: generalised number of true positives of target class.
            flat_target: the target tensor.
            wasserstein_distance_map: the map obtained from the above function.
        """
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)
        alpha_extended = alpha_extended.expand((flat_target.size(0), self.num_classes, flat_target.size(1)))
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)
        alpha_extended = torch.gather(alpha_extended, index=flat_target_extended, dim=1)

        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (2.0 - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_weights_generalized_true_positives(self, flat_target: torch.Tensor):
        """
        Args:
            flat_target: the target tensor.
        """
        one_hot = F.one_hot(flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
        volumes = torch.sum(one_hot, dim=2)
        alpha = 1.0 / (volumes + 1.0)
        return alpha


dice = Dice = DiceLoss
generalized_dice = GeneralizedDiceLoss
generalized_wasserstein_dice = GeneralizedWassersteinDiceLoss
