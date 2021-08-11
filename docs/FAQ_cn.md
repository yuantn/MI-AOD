# 常见问题解答

语言： 简体中文 | [English](FAQ.md)

这里列出了用户们遇到的一些常见问题，以及相应的解决方案。
如果您发现了任何经常出现的问题，欢迎充实本文档来帮助他人。
如果本文档不包括您的问题，欢迎在 [这里](../../../issues) 创建问题。
目前这里还没有包括开放中的问题，因为这些问题中有人可能会有进一步的疑问。

<!-- TOC -->

- [环境安装](#环境安装)
- [训练和测试](#训练和测试)
- [论文细节](#论文细节)
- [已修复错误和新功能](#已修复错误和新功能)

<!-- TOC -->

## 环境安装

1.  **问： 报错：`TypeError: forward() missing 1 required positional argument: 'x'`。（问题 [#3](../../../issues/3)、[#5](../../../issues/5)、[#15](../../../issues/15#issuecomment-854458413)）**
    
    **答：** 请参考 [修改 MMCV 包](installation_cn.md#修改-mmcv-包)。
    即如果你修改了 MMCV 包中的任何文件（包括但不限于：更新或重装 Python、Pytorch、MMDetection、MMCV、MMCV-FULL、conda 环境），
    你都应该将该代码库中提供的 `epoch_based_runner.py` 重新复制到 MMCV 的目录中去（如安装文档中所述）。

2.  **问： 报错：`AssertionError: MMCV==1.3.1 is used but incompatible. Please install mmcv>=1.0.5, <=1.0.5`。（问题 [#10](../../../issues/10)）**

    **答：** 请卸载 **mmcv** 和 **mmcv-full**，之后重装 **mmcv-full==1.0.5**。
    
3.  **问： 在安装 `mmcv==1.0.5` 后，仍然会报错：**

    ```bash
    ImportError: cannot import name 'Config' from 'mmcv' (unknown location)
    ModuleNotFoundError: No module named 'mmcv.utils'
    ```
    
    **（问题 [#13](../../../issues/13#issuecomment-841080219)）**
    
    **答：** 请按照 [这里](../../../blob/master/docs/installation_cn.md#环境安装) 的步骤 5 来安装搭建代码需要的包，并安装与编译 MMDetection 。
    

## 训练和测试

1.  **问： 报错：`AttributeError: 'Tensor' object has no attribute 'isnan'`。（问题 [#2](../../../issues/2) 和 [#9](../../../issues/9)）**

    **答：** 选项 1：根据 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v160)，重装 **Pytorch==1.6.0** 和 **TorchVision==0.7.0**。
    
    选项 2：检查报错 `AttributeError` 的位置，将 `if value.isnan()` 改为 `if value != value`（因为只有 nan != nan）。
    
    出错的位置应该位于 `./mmdet/models/dense_heads/MIAOD_head.py` 的第 483 和 569 行。
    
2.  **问： 在运行 `./script.sh 0` 时没有反应。（问题 [#6](../../../issues/6) 和 [#13](../../../issues/13)）**

    **答：** 当运行 `script.sh` 时，代码是在后台运行的。
    你可以通过在代码根目录下运行这个命令来查看输出日志：`vim log_nohup/nohup_0.log`。
    
    在 [另一章节](#已修复错误和新功能) 中，提供了另一个解决方案，可将日志直接输出到终端中。
    
3.  **问： 报错：`StopIteration`。（问题 [#7](../../../issues/7#issuecomment-823068004) 和 [#11](../../../issues/11)）**

    **答：** 感谢 [@KevinChow](https://github.com/kevinchow1993) 提供的解决方案。
    
    在 `mmdet/utils/active_datasets.py` 代码里的 `create_X_L_file()` 和 `create_X_U_file()` 函数中，在向 txt 文件写入之前，
    先让程序随机地 sleep 一段时间，让它们不在同时写入文件：
    
    ```python
        time.sleep(random.uniform(0,3))
        if not osp.exists(save_path):
            mmcv.mkdir_or_exist(save_folder)
            np.savetxt(save_path, ann[X_L_single], fmt='%s')
    ```

    在 `tools/train.py` 中调用 `create_X_L_file()` 和 `create_X_U_file()` 之后，加入如下代码来同步每个 GPU 上的线程：
    
    ```python
              if dist.is_initialized():
                  torch.distributed.barrier()
    ```
    
4.  **问： 我想在其他数据上运行 MI-AOD，我应该修改哪些文件？（问题 [#13](../../../issues/13#issuecomment-845709365)）**

    **答：** 如果你可以将你其他的训练和测试数据转换为 PASCAL VOC 格式的话，你只需要修改 `configs/MIAOD.py`。它包含了所有的参数和设置。
    
5.  **问： 验证时错误：`TypeError: 'DataContainer' object is not subscriptable`。（问题 [#14](../../../issues/14)）**

    **答：** 在 `mmdet/models/dense_heads/MIAOD_head.py` 文件的 `get_bboxes` 函数中，请将
    
    ```python
    img_shape = img_metas[img_id]['img_shape']
    ```
    
    修改为

    ```python
    img_shape = img_metas.data[0]
    ```

    注意：只有当你遇到这个问题时，你才需要做出修改。通常它不会在一个 GPU 环境中出现。

6.  **问： 在 `python tools/test.py $配置文件地址 $模型文件地址` 中，`$配置文件地址` 和 `$模型文件地址` 指的是什么？（问题 [#17](../../../issues/17)）**

    **答：** 有关这些参数的解释请参考 [这里](../../../blob/master/README_cn.md#训练和测试)，即：
    
    > 其中 `$配置文件地址` 应改为 `configs` 文件夹中的配置文件地址（通常应为 `configs/MIAOD.py`）
    
    > `$模型文件地址` 应改为训练后 `work_dirs` 文件夹中的模型文件（*.pth）的地址。

7.  **问： 当训练自定义数据集（前景类别数为 1）时，在训练过程中为何 `l_imgcls` 指标一直为 0 呢？（问题 [#23](../../../issues/23) 和 [#24](../../../issues/24)）**

    **答：** 为避免该情况，你可以在数据集中新建另一个类别，但那个类别没有对应的图像。


## 论文细节

1.  **问： 这个代码会被开源到 MMDetection 以更广泛地传播吗？（问题 [#1](../../../issues/1)）**

    **答：** MI-AOD 主要是为主动学习设计的，但是 MMDetection 更多地是为了目标检测。
    如果 MI-AOD 能够开源到主动学习的工具箱中将会更好。
    
2.  **问： 在最大化/最小化不确定性的顺序和其固定的网络层方面，论文和代码之间有些差别。（问题 [#4](../../../issues/4) 和 [#16](../../../issues/16#issuecomment-859363894)）**

    **答：** 我们的实验证明，如果最大化和最小化的步骤调换（包括固定的网络层），性能几乎不会变化。
    
3.  **问： 理论上，在论文的图 5 中，在初始已标注数据上的实验性能应该很相似。实验上为什么不是这样？（问题 [#4](../../../issues/4#issuecomment-800871469)）**

    **答：** 原因可以被总结为：
    - 对于未标注数据的有意使用
    - -> 已标注集和未标注集上对齐得更好的示例分布
    - -> 未标注集里的有效信息（预测差异）
    - -> 自然形成的无监督学习过程
    - -> 性能上的提升

4.  **问： 根据我的推导，如何保证已标注数据和未标注数据的分布偏差已经被最小化了？（问题 [#8](../../../issues/8)）**

    **答：** 你的推导的过程和结果都有一些问题。并且最小化分布偏差是通过两个步骤（最大化和最小化不确定性，如图 2（a）所示）实现的，不仅仅是最小化不确定性这一步。

5.  **问： 主动学习和半监督学习主要的区别是什么？我能否直接将主动学习用于半监督学习？（问题 [#12](../../../issues/12)）**

    **答：** 主动学习的核心是，我们首先用少量数据训练一个模型，然后计算不确定性（或其他设计的参数）来为下个主动学习周期选择信息量大的样本。
    然而，半监督学习设法以一种静态而不是动态的视角挖掘与利用未标注数据。
    
    我认为我们的工作 MI-AOD 巧妙地将半监督学习和主动学习结合了起来。即我们用半监督学习（或其核心思想）来在有限的已标注数据和足够的未标注数据下学习，
    并且使用主动学习来挑选信息量大的未标注数据并标记他们，此即为目前主动学习研究的趋势。当然将主动学习用于半监督学习中也是一个好的想法。
    
6.  **问： 在 `y_head_cls` 方面（论文中的公式（5）、`mmdet/dense_heads/MIAOD_retina_head.py` 代码中 `forward_single` 函数），论文和代码之间有些差别。代码中的 `maximum` 和 `softmax` 函数是什么意思？（问题 [#16](../../../issues/16)）**

    **答：** 代码中的公式为：
    
    ```python
    y_head_cls = y_head_f_mil.softmax(2) * y_head_cls_term2.sigmoid().max(2, keepdim=True)[0].softmax(1)
    ```
    
    它可以被简化为：
    
    ```python
    y_head_cls = A.softmax() * B.max().softmax()
    ```
    
    其中 A 和 B 是 MIL 头部和平均后的分类器头部的输出。
    
    `max(2, keepdim=True)[0]` 是为了突出具有最高分数的类别，这些类别最有可能被预测为前景。
    
    `softmax(x)` 的意思是 `exp(x)/sum_c(x)`，与论文中公式（5）一致。
    
7.  **问： 在计算不确定性中的分歧损失方面，论文和代码之间有些差别。（问题 [#16](../../../issues/16)）**

    **答：** 我们的实验表明，使用这两种损失的性能没有很大差别。

8.  **问： 为什么在论文的图 5（b）和（c）的第一个周期中，MI-AOD 和其他方法之间没有像（a）这样显著的差异？（问题 [#19](../../../issues/19)）**

    **答：** （a）中初始已标注样本的数量为 827，而（b）中为 1000。（a）中的训练时期（epoch）数为 26，而（b）中为 300，虽然 RetinaNet 检测网络在一定程度上领先于 SSD 检测网络。
     数据和训练时期越多，模型越拟合，MI-AOD 与其他方法的差异越小。
    
     类似地，（c）中有 2345 个初始已标注样本。并注意 MS COCO 是一个更具挑战性的数据集，所以在 2.0% 的已标注数据下，所有方法在早期学习周期中的表现都不那么令人满意，导致较低性能之间的差异很小。

9.  **问： 当在公式（6）中训练 MIL 分类器时，对于有多个类别的一张图像而言，如何获取整张图像的标签？（问题 [#20](../../../issues/20)）**

    **答：** 对于多类别目标图像的话，整张图像的标签将会是一个 1\*20 的 one-hot 张量（20 为 PASCAL VOC 的类别数）。
    在训练整体分类器的时候，每个类别的标签（即图像标签 [i]）也会各自分别的去训练。
    
    
## 已修复错误和新功能    
    
1.  **问： 在运行 `./script.sh 0` 时没有反应。（问题 [#6](../../../issues/6) 和 [#13](../../../issues/13)）**

    **答：** 如果你想直接在终端输出运行日志，请参考 [这里](../../../blob/master/README_cn.md#训练和测试)。
    
2.  **问： 报错：`AttributeError: 'NoneType' object has no attribute 'param_lambda'`。（问题 [#7](../../../issues/7)）**

    **答：** 该错误已被修复，请更新到最新版本。

3.  **问： 如果只在单机单卡上训练，还需要像 `script.sh` 和 `tools/dist_train.py` 那样分布式训练吗？（问题 [#15](../../../issues/15)）**

    **答：** 如果只在单机单卡上训练，请参考 [这里](../../../blob/master/README_cn.md#训练和测试)。
    
4.  **问： 报错：`AssertionError: Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"`。（问题 [#17](../../../issues/17)）**

    **答：** 该错误已被修复，请更新到最新版本。

5.  **问： 如何在 COCO 数据集上运行，如何修改 `active_datasets.py`？（问题 [#18](../../../issues/18)）**

    **答：** [这个代码库中](https://github.com/KapilM26/coco2VOC) 的代码可以用来将 COCO 数据集的 json 格式标注转为 PASCAL VOC 数据集的 xml 格式标注。COCO 数据集的 JPEG 格式图像可以直接被用作 PASCAL VOC 数据集的 JPEG 格式图像。
    
    如果使用这种方式，训练部分的代码大体不变，测试部分的代码可以直接用 mmdetection 中的配置文件替换掉。
    
    具体而言，[这里](../for_coco/README_cn.md) 已经准备好了 MS COCO 数据集上数据准备的说明。
    
6.  **问： 如何进行单张图像的推理测试（计算不确定性，或返回 bbox）？（问题 [#21](../../../issues/21) 和 [#22](../../../issues/22)）**

    **答：** 新功能已经更新，请参考 [这里](../../../blob/master/README_cn.md#训练和测试)。

