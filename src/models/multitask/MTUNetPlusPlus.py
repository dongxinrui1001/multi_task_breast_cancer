from typing import Sequence, Union
import logging
import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep


# U-Net++ 多任务网络，利用共享特征同时完成像素级分割和图像级分类。
class MTUNetPlusPlus(nn.Module):
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

        self.upcat_0_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_1 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_2_1 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

        self.upcat_0_2 = UpCat(
            spatial_dims, fea[1], fea[0] * 2, fea[0], act, norm, bias, dropout, upsample, halves=False
        )
        self.upcat_1_2 = UpCat(spatial_dims, fea[2], fea[1] * 2, fea[1], act, norm, bias, dropout, upsample)
        self.upcat_2_2 = UpCat(spatial_dims, fea[3], fea[2] * 2, fea[2], act, norm, bias, dropout, upsample)

        self.upcat_0_3 = UpCat(
            spatial_dims, fea[1], fea[0] * 3, fea[0], act, norm, bias, dropout, upsample, halves=False
        )
        self.upcat_1_3 = UpCat(spatial_dims, fea[2], fea[1] * 3, fea[1], act, norm, bias, dropout, upsample)

        self.upcat_0_4 = UpCat(
            spatial_dims, fea[1], fea[0] * 4, fea[5], act, norm, bias, dropout, upsample, halves=False
        )

        self.final_conv_0_1 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_2 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_3 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_4 = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

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
        x_0_1 = self.upcat_0_1(x_1_0, x_0_0)

        x_2_0 = self.conv_2_0(x_1_0)
        x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
        x_0_2 = self.upcat_0_2(x_1_1, torch.cat([x_0_0, x_0_1], dim=1))

        x_3_0 = self.conv_3_0(x_2_0)
        x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
        x_1_2 = self.upcat_1_2(x_2_1, torch.cat([x_1_0, x_1_1], dim=1))
        x_0_3 = self.upcat_0_3(x_1_2, torch.cat([x_0_0, x_0_1, x_0_2], dim=1))

        x_4_0 = self.conv_4_0(x_3_0)
        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
        x_2_2 = self.upcat_2_2(x_3_1, torch.cat([x_2_0, x_2_1], dim=1))
        x_1_3 = self.upcat_1_3(x_2_2, torch.cat([x_1_0, x_1_1, x_1_2], dim=1))
        x_0_4 = self.upcat_0_4(x_1_3, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))

        output_0_1 = self.final_conv_0_1(x_0_1)
        output_0_2 = self.final_conv_0_2(x_0_2)
        output_0_3 = self.final_conv_0_3(x_0_3)
        output_0_4 = self.final_conv_0_4(x_0_4)

        """
        Classifier
        """
        features_extracted = torch.cat([self.process_level_3(x_3_0), x_4_0, self.process_level_3(x_3_1)], dim=1)
        predicted_class = self.classifier(features_extracted)

        if self.deep_supervision:
            output = [predicted_class], [output_0_1, output_0_2, output_0_3, output_0_4]
        else:
            output = predicted_class, output_0_4

        return output


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = MTUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, n_classes=2, deep_supervision=False)
    logits, segmentation = model(seq_input)
