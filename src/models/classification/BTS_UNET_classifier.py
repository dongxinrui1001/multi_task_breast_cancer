import torch
from torch import nn
from collections import OrderedDict


# 创建 3×3 卷积；stride 控制下采样，groups 控制分组卷积，dilation 控制膨胀范围。
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """ 2D convolution which uses a kernel size of 3"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


# 组合卷积、归一化和 LeakyReLU 激活。把重复出现的卷积小模块封装起来，输入形状为 (N,C,H,W)，输出空间尺寸通常不变。
class ConvInNormLeReLU(nn.Sequential):
    """
    This class stacks a 2D Convolution, Instance Normalization and Leaky ReLU layers.

    Params
    ******
        - in_channels: Number of input channels
        - out_channels: Number of output channels

    """

    # 初始化网络层和超参数；此阶段只搭建模块，不执行前向计算。
    def __init__(self, in_channels, out_channels):
        super(ConvInNormLeReLU, self).__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, out_channels)),
                    ('InNorm', nn.InstanceNorm2d(out_channels)),
                    ('LeReLU', nn.LeakyReLU(inplace=True))
                ]
            )
        )


# 一个编码/解码层级的卷积块，先提取特征，再把通道数调整为后续层需要的大小。
class LevelBlock(nn.Sequential):
    """
    This class stacks two blocks of ConvInNormLeReLU (2D Convolution, Instance Normalization and Leaky ReLU layers).

    Params
    ******
        - in_channels: Number of input channels
        - mid_channels: Number of channels between the first and the second block
        - out_channels: Number of output channels

    """

    # 初始化网络层和超参数；此阶段只搭建模块，不执行前向计算。
    def __init__(self, in_channels, mid_channels, out_channels):
        super(LevelBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvInNormLRelu1', ConvInNormLeReLU(in_channels, mid_channels)),
                    ('ConvInNormLRelu2', ConvInNormLeReLU(mid_channels, out_channels))
                ])
        )


# 基于 BTS U-Net 编码特征的分类网络，去掉像素级输出，改用全局池化和全连接分类头。
class BTSUNetClassifier(nn.Module):
    """
    This class implements a variation of 2D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "BTS U-Net Classifier"

    # 初始化网络层和超参数；此阶段只搭建模块，不执行前向计算。
    def __init__(self, sequences, classes, width, deep_supervision=False):
        super(BTSUNetClassifier, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]
        self.classes = 1 if classes == 2 else classes

        # Encoders
        self.encoder = nn.Sequential(
            LevelBlock(sequences, widths[0] // 2, widths[0]),
            nn.MaxPool2d(2, 2),
            LevelBlock(widths[0], widths[1] // 2, widths[1]),
            nn.MaxPool2d(2, 2),
            LevelBlock(widths[1], widths[2] // 2, widths[2]),
            nn.MaxPool2d(2, 2),
            LevelBlock(widths[2], widths[3] // 2, widths[3]),
            nn.MaxPool2d(2, 2),
            LevelBlock(widths[3], widths[3], widths[3]),
        )

        # Define classification decoder layers
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # Flatten output
            nn.Linear(widths[3] * 8 * 8, 256),  # Fully-connected layer
            nn.ReLU(),
            nn.Linear(256, self.classes)  # Classification output
        )

        self.weights_initialization()

    # 按代码指定的初始化策略初始化卷积、归一化层和线性层参数。
    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 执行一次前向传播；输入是批量图像张量，返回当前模型定义的 logits。
    # 注意：分类模型通常返回分类 logits，分割模型返回像素 logits；不要在模型内部提前 sigmoid/softmax，损失函数会处理。
    def forward(self, x):

        # Encoding phase
        encoded = self.encoder(x)

        # Forward pass through classification decoder
        classification = self.classifier(encoded)

        return classification


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    ground_truth = torch.ones(1, 1)

    model = BTSUNetClassifier(sequences=1, classes=1, width=6, deep_supervision=False)
    outputs = model(seq_input)

    print(outputs)
    print(ground_truth)
    assert ground_truth.shape == outputs.shape

    criterion = nn.BCEWithLogitsLoss()
    print(f"Cross-Entropy: {criterion(ground_truth, outputs)}")
