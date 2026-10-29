import logging
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset.BUSI_dataset import BUSI

warnings.filterwarnings("ignore")
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

# TODO: Update BUSI dataloader function
def BUSI_dataloader(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                    train_size=0.8, classes=None, path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/",
                    oversampling=True, semantic_segmentation=False):

    """构造一次普通的 train/validation/test 数据划分。

    ``path_images`` 必须指向包含 ``mapping.csv``、``images`` 和 ``masks`` 的目录。
    函数先按类别分层切分，再只对训练集做过采样，最后把三个 DataFrame 包装成
    ``BUSI`` Dataset 和 PyTorch DataLoader。返回值是
    ``(train_loader, val_loader, test_loader)``。
    """

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # mapping.csv 是样本索引表；Dataset 不直接扫描目录，而是依赖这里的路径和标签。
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # 先切出训练集，再把剩余部分平均切成验证集和测试集；stratify 保持类别比例。
    train_mapping, val_mapping_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True,
                                                   stratify=mapping['class'])
    val_mapping, test_mapping = train_test_split(val_mapping_, test_size=0.5, random_state=int(seed), shuffle=True,
                                                 stratify=val_mapping_['class'])

    if remove_outliers:
        train_mapping = filter_anomalous_cases(train_mapping)
        val_mapping = filter_anomalous_cases(val_mapping)
        test_mapping = filter_anomalous_cases(test_mapping)

    # 只复制训练集中的少数类，验证集和测试集保持真实分布，避免评估偏乐观。
    if oversampling:
        train_mapping_malignant = train_mapping[train_mapping['class'] == 'malignant']
        train_mapping = pd.concat([train_mapping, train_mapping_malignant])

    # logging datasets
    logging.info(train_mapping)
    logging.info(val_mapping)
    logging.info(test_mapping)

    # Creating the train-validation-test datasets
    train_dataset = BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations,
                         normalization=normalization, semantic_segmentation=semantic_segmentation)
    val_dataset = BUSI(mapping_file=val_mapping, transforms=None, augmentations=augmentations,
                       normalization=normalization, semantic_segmentation=semantic_segmentation)
    test_dataset = BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                        normalization=normalization, semantic_segmentation=semantic_segmentation)

    logging.info(f"Size of train dataset: {train_dataset.__len__()}")
    logging.info(f"Shape of images used for training: {train_dataset.__getitem__(0)['image'].shape}")
    logging.info(f"Size of validation dataset: {val_dataset.__len__()}")
    logging.info(f"Shape of images used for validating: {val_dataset.__getitem__(0)['image'].shape}")
    logging.info(f"Size of test dataset: {test_dataset.__len__()}")
    logging.info(f"Shape of images used for testing: {test_dataset.__getitem__(0)['image'].shape}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def BUSI_dataloader_CV(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                       train_size=0.8, classes=None, n_folds=5, oversampling=True, use_duplicated_to_train=False,
                       path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/", semantic_segmentation=False):
    """构造分层 K 折交叉验证所需的 DataLoader 列表。

    每一折先由 ``StratifiedKFold`` 产生测试集，再从剩余数据切出验证集。
    返回三个列表；列表下标对应同一个 fold：
    ``train_loaders[i]``、``val_loaders[i]``、``test_loaders[i]``。
    """

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    if use_duplicated_to_train:
        mapping = filter_incongruent_cases(mapping)
        mapping, mapping_out_complementary = filter_train_cases(mapping)

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # 每一折都保证测试集类别比例尽量接近总体分布。
    fold_trainset, fold_valset, fold_testset = [], [], []
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    for n, (train_ix, test_ix) in enumerate(kfold.split(mapping, mapping['class'])):
        train_val_mapping, test_mapping = mapping.iloc[train_ix], mapping.iloc[test_ix]
        test_mapping['fold'] = [n] * len(test_mapping)

        # 在当前 fold 的训练候选集里继续切分训练集和验证集。
        train_mapping, val_mapping = train_test_split(train_val_mapping, train_size=train_size, random_state=int(seed),
                                                      shuffle=True, stratify=train_val_mapping['class'])

        if remove_outliers:
            train_mapping = filter_anomalous_cases(train_mapping)
            val_mapping = filter_anomalous_cases(val_mapping)
            test_mapping = filter_anomalous_cases(test_mapping)

        if use_duplicated_to_train:
            logging.info(f"DD: {train_mapping.shape} {mapping_out_complementary.shape}")
            train_mapping = pd.concat([train_mapping, mapping_out_complementary])

        # 过采样只作用于当前 fold 的训练集，不能把验证/测试样本复制进去。
        if oversampling:
            # train_mapping = oversampling_BUSI(train_mapping, seed)
            train_mapping = deterministic_oversampling(train_mapping)

        if n == 0:  # for simplicity, just showing distribution for fold 0
            logging.info(f"\nClass distribution for train set:"
                         f"\n{train_mapping['class'].value_counts(normalize=True).reset_index()}")
            logging.info(f"\nClass distribution for validation set:"
                         f"\n{val_mapping['class'].value_counts(normalize=True).reset_index()}")
            logging.info(f"\nClass distribution for test set:"
                         f"\n{test_mapping['class'].value_counts(normalize=True).reset_index()}")
            logging.info(f"Train size: {train_mapping.shape}")
            logging.info(f"Validation size: {val_mapping.shape}")
            logging.info(f"Test size: {test_mapping.shape}")

        # append the corresponding subset to train-val-test sets for each CV
        fold_trainset.append(BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations,
                                  normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_valset.append(BUSI(mapping_file=val_mapping, transforms=None, augmentations=augmentations,
                                normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_testset.append(BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                                 normalization=normalization, semantic_segmentation=semantic_segmentation))

    # Creating a list of dataloaders. Each component of the list corresponds to a CV fold
    train_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_trainset]
    val_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_valset]
    test_loader = [DataLoader(fold, batch_size=1) for fold in fold_testset]

    return train_loader, val_loader, test_loader


def BUSI_dataloader_CV_prod(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                            train_size=0.8, classes=None, n_folds=5, oversampling=True, use_duplicated_to_train=False,
                            path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/", semantic_segmentation=False):
    """构造生产/最终训练流程使用的 K 折 DataLoader。

    与 ``BUSI_dataloader_CV`` 的主要区别是：这里把每一折的训练集和验证集
    合并后再训练，因此返回 ``(train_loaders, test_loaders)``，没有独立验证集。
    """

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    if use_duplicated_to_train:
        mapping = filter_incongruent_cases(mapping)
        mapping, mapping_out_complementary = filter_train_cases(mapping)

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # splitting dataset into train-val-test CV
    fold_trainset, fold_valset, fold_testset = [], [], []
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    for n, (train_ix, test_ix) in enumerate(kfold.split(mapping, mapping['class'])):
        train_val_mapping, test_mapping = mapping.iloc[train_ix], mapping.iloc[test_ix]
        test_mapping['fold'] = [n] * len(test_mapping)

        # Splitting the mapping dataset into train_mapping, val_mapping and test_mapping
        train_mapping, val_mapping = train_test_split(train_val_mapping, train_size=train_size, random_state=int(seed),
                                                      shuffle=True, stratify=train_val_mapping['class'])

        if remove_outliers:
            train_mapping = filter_anomalous_cases(train_mapping)
            val_mapping = filter_anomalous_cases(val_mapping)
            test_mapping = filter_anomalous_cases(test_mapping)

        if use_duplicated_to_train:
            train_mapping = pd.concat([train_mapping, mapping_out_complementary])

        if oversampling:
            # train_mapping = oversampling_BUSI(train_mapping, seed)
            train_mapping = deterministic_oversampling(train_mapping)

        train_mapping = pd.concat([train_mapping, val_mapping])
        if n == 0:  # for simplicity, just showing distribution for fold 0
            logging.info(f"\nClass distribution for train set:"
                         f"\n{train_mapping['class'].value_counts(normalize=True).reset_index()}")
            logging.info(f"\nClass distribution for test set:"
                         f"\n{test_mapping['class'].value_counts(normalize=True).reset_index()}")
            logging.info(f"Train size: {train_mapping.shape}")
            logging.info(f"Test size: {test_mapping.shape}")

        # append the corresponding subset to train-val-test sets for each CV
        fold_trainset.append(BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations,
                                  normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_testset.append(BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                                 normalization=normalization, semantic_segmentation=semantic_segmentation))

    # Creating a list of dataloaders. Each component of the list corresponds to a CV fold
    train_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_trainset]
    test_loader = [DataLoader(fold, batch_size=1) for fold in fold_testset]

    return train_loader, test_loader


def UCLM_dataloader(batch_size, path_images, augmentations=None, normalization=None, classes=None):
    """读取外部 UCLM 数据集并返回一个 DataLoader。

    该函数不做交叉验证，只把 mapping.csv 中指定类别的样本作为一个整体加载，
    主要用于外部数据测试或迁移实验。
    """

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    logging.info(f"\nClass distribution dataset: {mapping['class'].value_counts(normalize=True).reset_index()}")
    logging.info(f"Train size: {mapping.shape}")

    dataset = BUSI(mapping_file=mapping, transforms=None, augmentations=augmentations, normalization=normalization,
                   semantic_segmentation=None)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def filter_anomalous_cases(mapping):
    """按预先记录的类别和 ID 删除异常样本。

    这是基于人工整理的黑名单过滤，不会自动检测异常；如果数据集 ID 发生变化，
    这些硬编码 ID 可能不再对应原来的图像。
    """
    logging.info("Filtering anomalous cases")
    anomalous_cases = {
        'benign': [435, 433, 42, 131, 437, 269, 333, 399, 403, 406, 85, 164, 61, 94, 108, 114, 116, 119, 122, 201, 302,
                   394, 402, 199, 248, 242, 288, 236, 247, 233, 299, 4, 321, 25, 153],
        'malignant': [145, 51, 77, 78, 93, 94, 52, 106, 107, 18, 116],
        'normal': [34, 1]
    }

    for cls, ids in anomalous_cases.items():
        mapping = mapping[~((mapping['class'] == cls) & (mapping['id'].isin(ids)))]

    return mapping


def filter_incongruent_cases(mapping):
    """删除图像、类别或标注不一致的样本，供重复数据相关实验使用。"""
    logging.info("Filtering anomalous cases")
    mapping_out = mapping.copy()
    anomalous_cases = {
        'benign': [42, 131, 269, 333, 399, 406, 433, 437, 85, 164, 333],
        'malignant': [51, 52, 77, 78, 93, 94, 145, 51, 52],
        'normal': [1, 34]
    }

    for cls, ids in anomalous_cases.items():
        mapping_out = mapping_out[~((mapping_out['class'] == cls) & (mapping_out['id'].isin(ids)))]

    return mapping_out


def filter_train_cases(mapping):
    """把指定样本从主训练映射中移出，并返回被移出的互补子集。

    返回 ``(mapping_out, mapping_out_complementary)``；后者可在特定实验中重新
    拼回训练集，因此该函数本身不会删除磁盘文件。
    """
    logging.info("Filtering anomalous cases")
    mapping_out = mapping.copy()
    anomalous_cases = {
        'benign': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 25, 30, 33, 35, 37, 38, 44,
                   50, 51, 52, 58, 60, 62, 64, 65, 81, 86, 96, 99, 105, 110, 127, 128, 129, 130, 132, 133, 134, 135,
                   136, 138, 139, 140, 141, 150, 151, 152, 153, 154, 155, 156, 157, 158, 163, 177, 197, 199, 200, 201,
                   202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222,
                   223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
                   244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
                   264, 265, 266, 267, 268, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 284, 285,
                   287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306,
                   307, 308, 309, 310, 312, 316, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331,
                   332, 395, 396, 400, 404, 411, 412, 413, 415, 419, 421, 422, 423, 424, 425, 426],
        'malignant': [4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 34, 39, 42, 65, 66, 80, 81, 88, 92, 95, 96, 97, 98, 99,
                      106, 107, 109, 110, 111, 112, 114, 116, 118, 119, 123, 128, 129],
        'normal': [5, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 43, 44,
                   45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 81, 97,
                   98, 104, 107, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
    }

    for cls, ids in anomalous_cases.items():
        mapping_out = mapping_out[~((mapping_out['class'] == cls) & (mapping_out['id'].isin(ids)))]

    mapping_out_complementary = mapping.loc[~mapping.index.isin(mapping_out.index)]

    return mapping_out, mapping_out_complementary


def oversampling_BUSI(mapping_df, seed):
    """随机补齐少数类，使其数量接近 benign 类数量。

    malignant 使用不放回抽样，normal 使用可放回抽样。该函数保留在代码中，
    当前交叉验证流程实际使用的是下面的 ``deterministic_oversampling``。
    """
    n_ben = len(mapping_df[mapping_df['class'] == 'benign'])
    if 'malignant' in set(mapping_df['class']):
        n_mal = len(mapping_df[mapping_df['class'] == 'malignant'])
        extra_malignant_images = mapping_df[mapping_df['class'] == 'malignant'].sample(n=n_ben - n_mal, random_state=seed)
        mapping_df = pd.concat([mapping_df, extra_malignant_images])
    if 'normal' in set(mapping_df['class']):
        n_nor = len(mapping_df[mapping_df['class'] == 'normal'])
        extra_normal_images = mapping_df[mapping_df['class'] == 'normal'].sample(n=n_ben - n_nor, random_state=seed, replace=True)
        mapping_df = pd.concat([mapping_df, extra_normal_images])

    return mapping_df


def deterministic_oversampling(mapping_df):
    """按类别比例确定性复制训练样本。

    每类复制因子约为 ``round(1 / 类别比例)``，再把额外副本拼回原表。
    例如某类占 20%，复制因子约为 5，最终会增加 4 份该类数据。
    返回仍是 DataFrame，原始图像文件不会被复制。
    """

    def compute_scaling_factor(mapping_df):
        data = mapping_df["class"].value_counts(normalize=True).reset_index()
        data["scaling_factor"] = round(1 / data["class"], 0).astype(int)
        return dict(zip(data['index'], data["scaling_factor"]))

    scaling_factor = compute_scaling_factor(mapping_df)

    oversampled_dfs = []
    for class_name, factor in scaling_factor.items():
        if factor > 1:
            class_df = mapping_df[mapping_df['class'] == class_name]
            oversampled_dfs.append(pd.concat([class_df] * (factor-1)))
        else:
            class_df = mapping_df[mapping_df['class'] == class_name]
            oversampled_dfs.append(pd.concat([class_df]))

    mapping_df = pd.concat([mapping_df] + oversampled_dfs, ignore_index=True)

    return mapping_df


def load_datasets(config_training, config_data, transforms, mode='CV'):
    """根据配置选择数据加载模式，并统一转发参数。

    ``mode='CV'`` 返回训练、验证、测试三个 DataLoader 列表；
    ``mode='CV_PROD'`` 返回训练和测试列表；``mode='UCLM'`` 返回一个外部数据
    DataLoader。训练入口通常只调用这个函数，而不直接调用底层构造函数。
    """
    if mode == 'CV':
        train_loaders, val_loaders, test_loaders = BUSI_dataloader_CV(seed=config_training['seed'],
                                                                      batch_size=config_data['batch_size'],
                                                                      transforms=transforms,
                                                                      remove_outliers=config_data['remove_outliers'],
                                                                      train_size=config_data['train_size'],
                                                                      n_folds=config_training['CV'],
                                                                      augmentations=config_data['augmentation'],
                                                                      normalization=None,
                                                                      classes=config_data['classes'],
                                                                      oversampling=config_data['oversampling'],
                                                                      use_duplicated_to_train=config_data['use_duplicated_to_train'],
                                                                      path_images=config_data['input_img'])
        return train_loaders, val_loaders, test_loaders
    if mode == 'CV_PROD':
        train_loaders, test_loaders = BUSI_dataloader_CV_prod(seed=config_training['seed'],
                                                              batch_size=config_data['batch_size'],
                                                              transforms=transforms,
                                                              remove_outliers=config_data['remove_outliers'],
                                                              train_size=config_data['train_size'],
                                                              n_folds=config_training['CV'],
                                                              augmentations=config_data['augmentation'],
                                                              normalization=None,
                                                              classes=config_data['classes'],
                                                              oversampling=config_data['oversampling'],
                                                              path_images=config_data['input_img'])
        return train_loaders, test_loaders
    if mode == 'UCLM':
        dataloader = UCLM_dataloader(batch_size=1,
                                     path_images="/home/carlos/Documentos/proyectos/breast_cancer/Datasets/BUS_UCLM_postprocessed_128",
                                     augmentations=config_data['augmentation'],
                                     normalization=None,
                                     classes=config_data['classes'])
        return dataloader
