from __future__ import print_function, division

import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, hflip, vflip

from src.utils.images import min_max_scaler
from src.utils.custom_transforms import apply_SOBEL_filter


class BUSI(Dataset):
    """BUSI (Breast UltraSound Image) dataset.

    这个类把 ``mapping.csv`` 中的一行转换成 PyTorch 可以训练的样本。
    每个样本包含图像 ``image``、分割掩膜 ``mask``、类别 ``label``，以及
    用于追踪原始样本的 ID 和肿瘤尺寸信息。数据在初始化时读入内存，
    ``__getitem__`` 只负责取样本、变换和返回字典。
    """

    def __init__(
            self,
            mapping_file: pd.DataFrame,
            transforms=None,
            augmentations=None,
            normalization=None,
            semantic_segmentation=False
    ):
        """创建数据集。

        参数 ``mapping_file`` 必须是 pandas.DataFrame，并且至少包含
        ``img_path``、``mask_path``、``id``、``class``、``dim1``、``dim2``
        和 ``tumor_pixels`` 这些列。``transforms`` 通常是 torchvision 的
        随机变换序列；它必须同时作用于图像和 mask，才能保持像素对应。
        ``augmentations`` 控制额外生成的 CLAHE、Sobel、亮度和对比度通道。
        """
        super(BUSI, self).__init__()

        if augmentations is None:
            augmentations = {}

        self.mapping_file = mapping_file
        self.transforms = transforms
        self.semantic_segmentation = semantic_segmentation
        self.transforms_applied = {}
        # 只要增强配置中至少有一个 True，就会额外生成增强图像通道。
        self.augmentations = True if sum([v for k, v in augmentations.items()]) else False
        if augmentations:
            self.CLAHE = augmentations.get("CLAHE", False)
            self.SOBEL = augmentations.get("SOBEL", False)
            self.brightness_brighter = augmentations.get("brightness_brighter", False)
            self.brightness_darker = augmentations.get("brightness_darker", False)
            self.contrast_high = augmentations.get("contrast_high", False)
            self.contrast_low = augmentations.get("contrast_low", False)

        self.normalization = normalization

        # 预先把图片、掩膜和元数据读入列表；这样训练时不再反复访问磁盘。
        self.data = []
        for index, row in self.mapping_file.iterrows():
            # OpenCV 以灰度模式读取超声图像。结果形状通常是 (H, W)。
            image = cv2.imread(row['img_path'], 0)
            if semantic_segmentation:
                mask = cv2.imread(row['mask_path'], 1).transpose((2, 0, 1))
            else:
                mask = cv2.imread(row['mask_path'], 0)
                mask[mask == 255] = 1

            # 读取 mapping.csv 中的元数据，后面会原样放回返回字典。
            patient_id = row['id']
            class_ = row['class']
            dim1 = row['dim1']
            dim2 = row['dim2']
            tumor_pixels = row['tumor_pixels']
            if self.semantic_segmentation:
                if class_ == 'benign':
                    label = torch.ones(1)
                elif class_ == 'normal':
                    label = torch.zeros(1)
                elif class_ == 'malignant':
                    label = 2 * torch.ones(1)
                else:
                    raise Exception(f"\n\t-> Unknown class: {row['class']}")
            else:
                if class_ == 'malignant':
                    label = torch.ones(1)
                elif class_ == 'benign':
                    label = torch.zeros(1)
                elif class_ == 'normal':
                    label = 2 * torch.ones(1)
                else:
                    raise Exception(f"\n\t-> Unknown class: {row['class']}")

            # 每一行变成一个字典；Dataset 的 __getitem__ 会按 idx 取出它。
            self.data.append({
                'patient_id': patient_id,
                'label': label,
                'class_': class_,
                'image': image,
                'mask': mask,
                'dim1': dim1,
                'dim2': dim2,
                'tumor_pixels': tumor_pixels
            })

    def __len__(self):
        """返回样本数量，供 DataLoader 决定一个 epoch 有多少样本。"""
        return len(self.data)

    def __getitem__(self, idx):
        """按索引返回一个训练样本。

        返回字典中的图像形状通常是 ``(C, H, W)``，mask 是 ``(1, H, W)``，
        label 是保存为单元素张量的类别编号。这里把 mask 和 image 拼到
        一起再做几何变换，保证旋转或翻转后两者仍然严格对齐。
        """

        patient_info = self.data[idx]

        # 原始灰度图是 (H, W)，神经网络要求带通道维，因此变成 (1, H, W)。
        image = torch.unsqueeze(torch.as_tensor(patient_info['image'], dtype=torch.float32), 0)
        mask = torch.as_tensor(patient_info['mask'], dtype=torch.float32)
        if not self.semantic_segmentation:
            mask = torch.unsqueeze(mask, 0)

        if self.normalization is not None:
            image = min_max_scaler(image)

        # 这些增强不是随机几何变换，而是额外生成输入通道。
        aumengs = []
        if self.augmentations and not self.semantic_segmentation:

            if self.CLAHE:
                clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
                aumengs.append(torch.unsqueeze(torch.as_tensor(clahe.apply(patient_info['image']),
                                                               dtype=torch.float32), 0))

            if self.SOBEL:
                aumengs.append(torch.unsqueeze(torch.as_tensor(apply_SOBEL_filter(patient_info['image']),
                                                               dtype=torch.float32), 0))

            if self.brightness_brighter:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_brighter = cv2.add(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_brighter, dtype=torch.float32), 0))
            if self.brightness_darker:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_darker = cv2.subtract(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_darker, dtype=torch.float32), 0))

            if self.contrast_low:  # contrast
                matrix1 = np.ones(patient_info['image'].shape) * .02
                img_low_contrast = np.uint8(cv2.multiply(np.float64(patient_info['image']), matrix1))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_low_contrast, dtype=torch.float32), 0))
            if self.contrast_high:  # contrast
                matrix2 = np.ones(patient_info['image'].shape) * 1.5
                img_high_contrast = np.uint8(np.clip(cv2.multiply(np.float64(patient_info['image']), matrix2), 0, 255))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_high_contrast, dtype=torch.float32), 0))

        # 没有额外增强通道时，将 mask 和 image 拼接后统一做几何变换。
        if self.transforms is not None and not self.augmentations:
            joined = self.transforms(torch.cat([mask, image], dim=0))
            # joined, self.transforms_applied = apply_transformations(torch.cat([mask, image], dim=0), self.transforms)
            if not self.semantic_segmentation:
                mask = torch.unsqueeze(joined[0, :, :], 0)
                image = torch.unsqueeze(joined[1, :, :], 0)
            else:
                mask = joined[0:-1, :, :]
                image = torch.unsqueeze(joined[-1, :, :], 0)

        # 有增强通道时，所有通道一起变换，保证每个增强图像与 mask 对齐。
        if self.transforms is not None and self.augmentations and not self.semantic_segmentation:
            joined = torch.cat([mask, image] + aumengs, dim=0)
            joined = self.transforms(joined)
            # joined, self.transforms_applied = apply_transformations(joined, self.transforms)
            mask = torch.unsqueeze(joined[0, :, :], 0)
            image = joined[1:, :, :]

        # 如果只要求额外增强、不要求随机旋转/翻转，就直接拼接通道。
        if self.transforms is None and self.augmentations and not self.semantic_segmentation:
            image = torch.cat([image] + aumengs, dim=0)

        # if self.normalization is not None:
        #     # image = torch.cat([image, aug1], dim=0)
        #     image = min_max_scaler(image)

        return {
            'patient_id': patient_info['patient_id'],
            'label': patient_info['label'],
            'class': patient_info['class_'],
            'image': image,
            'mask': mask,
            'dim1': patient_info['dim1'],
            'dim2': patient_info['dim2'],
            'tumor_pixels': patient_info['tumor_pixels'],
            # 'transforms_applied': self.transforms_applied
        }


def testing_apply_transformations(image, transforms_sequential):
    """手动执行随机翻转和旋转，并记录实际应用了哪些变换。

    该函数返回 ``(变换后的图像, 变换记录字典)``。例如返回的记录可能是
    ``{'horizontal_flip': True, 'vertical_flip': False, 'rotation': 90}``。
    当前训练主流程使用 torchvision 的 transforms；这个函数主要用于测试
    或需要显式记录增强操作的场景。
    """

    # This will store the transformations applied
    transforms_applied = {'horizontal_flip': False, 'vertical_flip': False, 'rotation': 0}

    # Random horizontal flips
    if random.random() < transforms_sequential.get('horizontal_flip') != .0:
        transforms_applied['horizontal_flip'] = True
        image = hflip(image)

    # Random vertical flips
    if random.random() < transforms_sequential.get('vertical_flip') != .0:
        transforms_applied['vertical_flip'] = True
        image = vflip(image)

    # Random rotations between 0-360 degrees
    if random.random() < transforms_sequential.get('rotation'):
        # angle = random.randint(0, 360)
        angle = int(np.random.choice(range(0, 360)))
        transforms_applied['rotation'] = angle
        image = rotate(image, angle)

    return image, transforms_applied
