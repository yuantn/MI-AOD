# 配置文件 MIAOD.py 中的变量与参数

语言：简体中文 | [English](README.md)

<!-- TOC -->

- [基础文件](#基础文件)
- [数据集](#数据集)
- [模型训练](#模型训练)
- [结果](#结果)
- [MI-AOD 的特有设定](#mi-aod-的特有设定)

<!-- TOC -->

# 基础文件

- **\_base\_**（第 4-7 行）：基本的配置文件，包括 RetinaNet、PASCAL VOC 和一些运行设定。

## 数据集

- **data_root**（第 2 行）：你实际的目录。在 `_base_/voc0712.py` 中有一个同样的 **data_root** 需要修改。

- **data**（第 9-16 行）：用于训练、验证、测试的数据。

  - **test**（第 10 行）：用于测试的数据。
  
    - **ann_file**（第 11-14 行）：标注文件的路径。
    
    - **img_prefix**（第 15 行）：图像的路径。

## 模型训练Model Training

- **model**（第 17 行）：使用的模型，包括基网、模型的颈部与头部。

  - **bbox_head**：模型的边界框头部。
  
    - **C**：数据集中类别数量。
    
- **optimizer**（第 19 行）：使用的优化器，包括学习率、动量因子和权重衰减因子。

  - **lr**：学习率。
  
  - **momentum**：动量参数。
  
  - **weight_decay**：权重衰减参数。
  
- **optimizer_config**（第 20 行）：梯度均衡参数。

- **lr_config**（第 22 行）：学习率的设置。

  - **step**：学习率下降的时刻。
  
## 结果

- **checkpoint_config**（第 24 行）：保存模型的频率。

- **log_config**（第 26 行）：打印训练日志文件的频率。

- **evaluation**（第 31 行）：评价模型的频率。

- **work_directory**（第 48 行）：保存日志和文件的工作目录。要获取更多信息，请参考 [这里](../README_cn.md#结果)。

## MI-AOD 的特有设定

- **epoch_ratio**（第 29 行）：已标注集训练步骤和重加权与最大/最小化示例不确定性步骤训练的时期（Epoch）数。

- **epoch**（第 33 行）：外层循环的数量（即除了第一次已标注集训练步骤之外的其他 3 个训练步骤的数量）。

- **X_L_repeat**（第 36 行）：已标注集数据重复的次数。

- **X_U_repeat**（第 37 行）：未标注集数据重复的次数。

- **train_cfg**（第 39 行）：模型训练的一些参数。

  - **param_lambda**：正则化超参数 ![lambda](http://latex.codecogs.com/gif.latex?\bg_white\lambda)。

- **k**（第 40 行）：超参数 _k_，样本选择中在一个未标注图像中选择观察的最高若干示例不确定性的数量。

- **X_S_size**（第 43 行）：![X_S](http://latex.codecogs.com/gif.latex?\bg_white\mathit{X}_S)，新选出集合的大小。

- **X_L_0_size**（第 44 行）：![X_L^0](http://latex.codecogs.com/gif.latex?\bg_white\mathit{X}_L^0)，初始已标注集的大小。

- **cycles**（第 46 行）：主动学习周期的数量。
