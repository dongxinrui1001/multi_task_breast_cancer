from pathlib import Path
import pandas as pd

# 当前脚本位于 src 下，因此自动定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "20261030_171323_MTnnUNet_24_alpha_0.35_batch_2_benign_malignant_normal"
)

# 读取 4 个 fold 的结果
csv_files = sorted(
    RUN_DIR.glob("fold_*/results_segmentation.csv")
)

if not csv_files:
    raise FileNotFoundError(
        f"没有找到 results_segmentation.csv，查找目录为：{RUN_DIR}"
    )

all_results = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # 记录当前结果来自哪个 fold
    df["fold"] = csv_file.parent.name

    all_results.append(df)

# 合并所有 fold
results = pd.concat(all_results, ignore_index=True)

print("class 列的实际内容：")
print(results["class"].unique())

# 如果 class 是数字，取消下面注释
# results["class"] = results["class"].map({
#     0: "benign",
#     1: "malignant",
#     2: "normal"
# })

# 第一步：每个 fold、每个类别计算平均 Dice
fold_class_dice = (
    results
    .groupby(["fold", "class"])["DICE"]
    .mean()
    .reset_index()
)

# 第二步：对 4 个 fold 的类别平均 Dice 求均值和标准差
paper_style_dice = (
    fold_class_dice
    .groupby("class")["DICE"]
    .agg(["mean", "std", "count"])
)

print("\n各类别 Dice：")
print(paper_style_dice)

# 三个类别 DSC Average
class_mean_dice = paper_style_dice["mean"].mean()

print("\n三个类别 Dice 的平均值：")
print(class_mean_dice)

# 保存结果
output_file = RUN_DIR / "segmentation_dice_by_class.csv"
paper_style_dice.to_csv(output_file, encoding="utf-8-sig")

print(f"\n统计结果已保存到：{output_file}")