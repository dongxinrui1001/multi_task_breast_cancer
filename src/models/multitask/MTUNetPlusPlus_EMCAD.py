from typing import Sequence, Union
import logging
import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep

from src.models.decoders.emcad import EMCAD


# 解码器分割分支变为EMCAD，分类不变
class MTUNetPlusPlus_EMCAD(nn.Module):
    # 初始化网络层和超参数；此阶段只搭建模块，不执行前向计算。
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        n_classes: int = 3,
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        deep_supervision: bool = False,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        """
        An Extension of UNet++ implementation with 1D/2D/3D supports. It classifies

        Based on MONAI library and reference:

            Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image
            Segmentation". 4th Deep Learning in Medical Image Analysis (DLMIA)
            Workshop, DOI: https://doi.org/10.48550/arXiv.1807.10165

        """
        super().__init__()

        self.deep_supervision = deep_supervision
        self.n_classes = n_classes
        if n_classes == 2:
            self.n_classes = 1

        fea = ensure_tuple_rep(features, 6)\

        logging.info(f"BasicUNetPlusPlus features: {fea}.")

        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

        # Define classification decoder layers
        self.process_level_3 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.classifier = nn.Sequential(
            TwoConv(spatial_dims, fea[4] * 3, 512, act, norm, bias, dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # Flatten output
            nn.Linear(512, 256),  # Fully-connected layer
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.n_classes)  # Classification output
        )
        #join emcad decoder
        self.emcad = EMCAD(
            channels=[fea[4], fea[3], fea[2], fea[1]],
            kernel_sizes=[1, 3, 5],
            expansion_factor=6
        )
        #add segmentation head
        self.emcad_seg_head_4 = nn.Conv2d(
            in_channels=fea[4],
            out_channels=out_channels,
            kernel_size=1
        )

        self.emcad_seg_head_3 = nn.Conv2d(
            in_channels=fea[3],
            out_channels=out_channels,
            kernel_size=1
        )

        self.emcad_seg_head_2 = nn.Conv2d(
            in_channels=fea[2],
            out_channels=out_channels,
            kernel_size=1
        )

        self.emcad_seg_head_1 = nn.Conv2d(
            in_channels=fea[1],
            out_channels=out_channels,
            kernel_size=1
        )

    # 执行一次前向传播；输入是批量图像张量，返回当前模型定义的 logits。
    # 注意：分类模型通常返回分类 logits，分割模型返回像素 logits；不要在模型内部提前 sigmoid/softmax，损失函数会处理。
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(x_0_0)
        x_2_0 = self.conv_2_0(x_1_0)
        x_3_0 = self.conv_3_0(x_2_0)
        x_4_0 = self.conv_4_0(x_3_0)

        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)


        """
        Classifier
        """
        features_extracted = torch.cat([self.process_level_3(x_3_0), x_4_0, self.process_level_3(x_3_1)], dim=1)
        predicted_class = self.classifier(features_extracted)

        '''
        segmentation
        '''
        d4, d3, d2, d1 = self.emcad(
            x_4_0,
            [x_3_0, x_2_0, x_1_0]
        )

        segmentation_4 = self.emcad_seg_head_4(d4)
        segmentation_3 = self.emcad_seg_head_3(d3)
        segmentation_2 = self.emcad_seg_head_2(d2)
        segmentation_1 = self.emcad_seg_head_1(d1)

        segmentation_4 = torch.nn.functional.interpolate(
            segmentation_4,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        segmentation_3 = torch.nn.functional.interpolate(
            segmentation_3,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        segmentation_2 = torch.nn.functional.interpolate(
            segmentation_2,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        segmentation_1 = torch.nn.functional.interpolate(
            segmentation_1,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        if self.deep_supervision:
            return [predicted_class], [
                segmentation_4,
                segmentation_3,
                segmentation_2,
                segmentation_1
            ]
        else:
            return predicted_class, segmentation_1


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = MTUNetPlusPlus_EMCAD(spatial_dims=2, in_channels=1, out_channels=1, n_classes=2, deep_supervision=False)
    logits, segmentation = model(seq_input)
