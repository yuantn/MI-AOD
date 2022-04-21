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
- [自定义修改](#自定义修改)

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
    
    **答：** 请按照 [这里](installation_cn.md#环境安装) 的步骤 5 来安装搭建代码需要的包，并安装与编译 MMDetection 。

4.  **问： 模型已经正常训练了一个循环，也生成了 `*.npy` 的权重文件后，在进入下一个循环时突然报错：`RuntimeError: CUDA error: no kernel image is available for execution on the device`。（问题 [#36](../../../issues/36)）**

    **答：** 训练过程并没有开始，`*0.npy` 是在第一次循环训练之前生成的。报错的原因是 mmdet 中 CUDA runtime version 和 compiler version 不匹配。

5.  **问： 报错：`ModuleNotFoundError: No module named 'torchvision.models.segmentation.data_loader'`。（问题 [#37](../../../issues/37)）**

    **答：** 请重新安装适配于你当前 CUDA 版本的 PyTorch 和 torchvision 。


## 训练和测试

1.  **问： 报错：`AttributeError: 'Tensor' object has no attribute 'isnan'`。（问题 [#2](../../../issues/2) 和 [#9](../../../issues/9)）**

    **答：** 选项 1：根据 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v160)，重装 **Pytorch==1.6.0** 和 **TorchVision==0.7.0**。
    
    选项 2：检查报错 `AttributeError` 的位置，将 `if value.isnan()` 改为 `if value != value`（因为只有 nan != nan）。
    
    出错的位置应该位于 `./mmdet/models/dense_heads/MIAOD_head.py` 的第 483 和 569 行。
    
2.  **问： 在运行 `./script.sh 0` 时没有反应。（问题 [#6](../../../issues/6) 和 [#13](../../../issues/13)）**

    **答：** 当运行 `script.sh` 时，代码是在后台运行的。
    你可以通过在代码根目录下运行这个命令来查看输出日志：`vim log_nohup/nohup_0.log`。
    
    在 [另一章节](#已修复错误和新功能) 中，提供了另一个解决方案，可将日志直接输出到终端中。
    
3.  **问： 报错：`StopIteration`。（问题 [#7](../../../issues/7#issuecomment-823068004)、[#11](../../../issues/11) 和 [#31](../../../issues/31)）**

    **答：** __如果使用单 GPU 训练：__
    
    请增加训练数据的数量。我们建议在 PASCAL VOC 数据集上用 RetinaNet 使用至少 5% 的图像（16551 * 5% = 827 张图像）。
    
    __如果使用多 GPU 训练：__
    
    感谢 [@KevinChow](https://github.com/kevinchow1993) 提供的解决方案。
    
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
    
4.  **问： 验证时错误：`TypeError: 'DataContainer' object is not subscriptable`。（问题 [#14](../../../issues/14)）**

    **答：** 在 `mmdet/models/dense_heads/MIAOD_head.py` 文件的 `get_bboxes` 函数中，请将
    
    ```python
    img_shape = img_metas[img_id]['img_shape']
    ```
    
    修改为

    ```python
    img_shape = img_metas.data[0]
    ```

    注意：只有当你遇到这个问题时，你才需要做出修改。通常它不会在一个 GPU 环境中出现。

5.  **问： 在 `python tools/test.py $配置文件地址 $模型文件地址` 中，`$配置文件地址` 和 `$模型文件地址` 指的是什么？（问题 [#17](../../../issues/17)）**

    **答：** 有关这些参数的解释请参考 [这里](../README_cn.md#训练和测试)，即：
    
    > 其中 `$配置文件地址` 应改为 `configs` 文件夹中的配置文件地址（通常应为 `configs/MIAOD.py`）
    
    > `$模型文件地址` 应改为训练后 `work_dirs` 文件夹中的模型文件（*.pth）的地址。
    
6.  **问： 在 `tools/train.py` 中，代码中是否首先在有标注数据集上训练，其目的是什么？（问题 [#26](../../../issues/23)）**

    **答：** 需要让第一次和最后一次迭代都是在有标注数据集上训练，以保证训练模型的稳定性。

7.  **问： 为什么对于未标注集，在 `epoch_based_runner.py` 的第 70-74 行也涉及到了对 GT 信息的操作（即 `gt_bboxes` 和 `gt_labels`）？（问题 [#28](../../../issues/28) 和 [#29](../../../issues/29)）**

    **答：** 这几行的操作是为了抹除未标注集图像的定位信息，这样在计算涉及到未标注集的损失时，就可以知道数据来源从而不反传梯度，如此实际上是没有使用其 GT 信息的。

8.  **问： `configs/MIAOD.py` 中的 `epoch_ratio = [3, 1]` 是什么意思？我可以将其更改为 `epoch_ratio = [3, 0]` 吗？（问题 [#31](../../../issues/31#issuecomment-881190530)）**

    **答：** 有关配置文件的说明请参考 [这里](../configs/README_cn.md#mi-aod-的特有设定)。
    
    如果将其更改为 [3, 0]，则不会存在最大化和最小化不确定性。

9.  **问： 报错：`IndexError: index 0 is out of bounds for Dimension 0 with size 0`。（问题 [#31](../../../issues/31#issuecomment-881223658)、[#39](../../../issues/39)、[#40](../../../issues/40)）**

    **答：** 一个可能的解决方案是：将 `mmdet/models/dense_heads/MIAOD_head.py` 中 `L_wave_min` 的第 479 行的
    
    ```python
    if y_loc_img[0][0][0] < 0：
    ```

    改为

    ```python
    if y_loc_img[0][0] < 0：
    ```
    
    如果不行，请仅在会报错时在错误行插入异常检测或使用 PyCharm 之类的 IDE 设置断点，并打印 `y_loc_img[0][0][0]` 和 `y_loc_img[0][0]` ，确认 `y_loc_img` 是否为空列表。
    
    如果是，请重新准备数据集的标注信息。
    
    如果是在自定义的数据集上进行训练，请参考 [自定义修改中问题 3](#自定义修改) 的注意事项。

10. **问： 如何保存每个 cycle 的训练模型？（问题 [#32](../../../issues/32)）**

    **答：** 目前这套代码可以保存每个 cycle 的训练模型，[这里](../README_cn.md#结果) 提供了一个示例输出文件夹的网盘链接。

11. **问： 在使用 `tools/test.py` 进行测试时，是否需要将 `config` 中的 `data.test.ann_file` 改为真正的测试集（而不是用 _trainval_ 的数据来计算不确定度）？（问题 [#32](../../../issues/32#issuecomment-879984647)）**

    **答：** 不是的，在这个代码中，我们测试时使用的是 _test_ 集，但在 `config` 中使用的是 `data.val` 的部分。请参见 [这里](../configs/_base_/voc0712.py)。

12. **问： `y_loc_img[0][0][0] < 0` 的意思是什么？（问题 [#40](../../../issues/40)）**

    **答：** 它的意思是当前批数据是未标注的数据，因为我们已经在 `epoch_based_runner.py` 中第 70-74 行将未标注数据的边界框的所有坐标设置为了 -1。
    
    此外，感谢 [@horadrim-coder](https://github.com/horadrim-coder) 提出的一种可避免报错 `IndexError: index 0 is out of bounds for dimension 0 with size 0` 的替代解决方案：
    
    1. 在 `epoch_based_runner.py` 中加入如下方法：
    
    ```python
    def _add_dataset_flag(self, X, is_unlabeled):
        or _img_meta in X['img_metas'].data[0]:
         _img_meta.update({'is_unlabeled': is_unlabeled})
    ```

    2. 在 `epoch_based_runner.py` 中的如下行加入代码：

    ```python
    第 31 行：`self._add_dataset_flag(X_L, is_unlabeled=False)`
    第 60 行：`self._add_dataset_flag(X_L, is_unlabeled=False)`
    第 79 行：`self._add_dataset_flag(X_U, is_unlabeled=True)`
    ```

    3. 在 `MIAOD_head.py` 中将 `y_loc_img[0][0][0] < 0` 替换为 `img_metas[0]['is_unlabeled']` （如第 479 和第 565 行）。

13. **问： 不能下载预训练的 SSD 模型。（问题 [#42](../../../issues/42)）**

    **答：** SSD的预训练模型 [链接](https://download.openmmlab.com/pretrain/third_party/vgg16_caffe-292e1171.pth) 可在 [最新版本的 mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json#L2) 中获取，下载后可在下载文件夹中将其移动到预训练模型的默认缓存文件夹中：
    
    ```bash
    mv vgg16_caffe-292e1171.pth ~/.cache/torch/hub/checkpoints
    ```


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

10. **问： 您能否分享图 5（b）中用于 MI-AOD 的确切数字（平均值和标准差）？（问题 [#25](../../../issues/25)）**

    **答：** 具体数字如下。
    
    |已标注图像数量|1k|2k|3k|4k|5k|6k|7k|8k|9k|10k|
    |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
    |MI-AOD 性能（%）|53.62|62.86|66.83|69.33|70.80|72.21|72.84|73.74|74.18|74.91|
    |MI-AOD 标准差（%）| 2.15 | 1.52 | 0.77 | 0.54 | 0.34 | 0.28 | 0.23 | 0.21 | 0.18 | 0.17 |

11. **问： 使用 SSD 网络对应的预训练模型是什么？（问题 [#33](../../../issues/33)）**

    **答：** 模型为 `open-mmlab` 提供的 `vgg16_caffe` 模型。

12. **问： SSD 的代码的 epoch 数与文章中阐明的 100 不一致。（问题 [#33](../../../issues/33#issuecomment-895883958)）**

    **答：** 我在论文中从未声明 SSD 的 epoch 数为 100，反而是 300（240+60，如论文 4.1 节所述）。

13. **问： 对于 Random 的结果，实验设置是怎样的？（问题 [#33](../../../issues/33#issuecomment-895883958)）**

    **答：** 我去掉了所有在未标注集合上训练的过程，并以随机的方式挑选图像得出的结果。

14. **问： 根据输出日志 `...Epoch [1][50/2000]...`，已标注图像的数量每个周期增加 2k（不是论文中的 1k）。（问题 [#38](../../../issues/38)）**

    **答：** 输出日志中的 `2000` 为 `X_L_0_size * X_L_repeat / samples_per_gpu = 1000 * 16 / 8`，添加的已标注图像数量应该是输出目录中 `X_L_0.npy` 的尺寸，即 `(1000,)`。

15. **问： 论文中的热力图是如何画出来的？（问题 [#41](../../../issues/41)）**

    **答：** 我们分别计算了两个分类器在每一个锚框上的 `l_dis`、`{y^}^cls`、`{l~}_dis`，并将这些值对应的锚框用热力图颜色填充。
    
    值越大越偏向红色，越小越偏向蓝紫色。
    
    最后我们将多个锚框叠加起的热力图与原图按一定比例相加，即可得到论文中的图 6。


## 已修复错误和新功能    
    
1.  **问： 在运行 `./script.sh 0` 时没有反应。（问题 [#6](../../../issues/6) 和 [#13](../../../issues/13)）**

    **答：** 如果你想直接在终端输出运行日志，请参考 [这里](../README_cn.md#训练和测试)。
    
2.  **问： 报错：`AttributeError: 'NoneType' object has no attribute 'param_lambda'`。（问题 [#7](../../../issues/7)）**

    **答：** 该错误已被修复，请更新到最新版本。

3.  **问： 如果只在单机单卡上训练，还需要像 `script.sh` 和 `tools/dist_train.py` 那样分布式训练吗？（问题 [#15](../../../issues/15)）**

    **答：** 如果只在单机单卡上训练，请参考 [这里](../README_cn.md#训练和测试)。
    
4.  **问： 报错：`AssertionError: Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"`。（问题 [#17](../../../issues/17)）**

    **答：** 该错误已被修复，请更新到最新版本。

5.  **问： 如何在 COCO 数据集上运行，如何修改 `active_datasets.py`？（问题 [#18](../../../issues/18) 和 [#27](../../../issues/27)）**

    **答：** [这个代码库中](https://github.com/KapilM26/coco2VOC) 的代码可以用来将 COCO 数据集的 json 格式标注转为 PASCAL VOC 数据集的 xml 格式标注。COCO 数据集的 JPEG 格式图像可以直接被用作 PASCAL VOC 数据集的 JPEG 格式图像。
    
    如果使用这种方式，训练部分的代码大体不变，测试部分的代码可以直接用 mmdetection 中的配置文件替换掉。
    
    具体而言，[这里](../for_coco/README_cn.md) 已经准备好了 MS COCO 数据集上数据准备的说明。
    
6.  **问： 如何进行单张图像的推理测试（计算不确定性，或返回 bbox）？（问题 [#21](../../../issues/21) 和 [#22](../../../issues/22)）**

    **答：** 新功能已经更新，请参考 [这里](../README_cn.md#训练和测试)。

7.  **问： 是否有提供使用 SSD 网络的代码？（问题 [#33](../../../issues/33)）**

    **答：** 该代码已经更新，关于使用 SSD 检测器的说明请参考 [这里](../README_cn.md#数据集准备)。


## 自定义修改

1.  **问： 我想在其他数据上运行 MI-AOD，我应该修改哪些文件？（问题 [#13](../../../issues/13#issuecomment-845709365)）**

    **答：** 如果你可以将你其他的训练和测试数据转换为 PASCAL VOC 格式的话，你只需要修改 `configs/MIAOD.py`。它包含了所有的参数和设置。

2.  **问： 当训练自定义数据集（前景类别数为 1）时，在训练过程中为何 `l_imgcls` 指标一直为 0 呢？（问题 [#23](../../../issues/23)、[#24](../../../issues/24)、[#34](../../../issues/34)、[#35](../../../issues/35)）**

    **答：** 为避免该情况，你可以在数据集中新建另一个类别，但那个类别没有对应的图像。

3.  **问： 如果使用完全无标注的数据作为未标注集，需要做哪些修改？（问题 [#29](../../../issues/29#issuecomment-871210792)）**

    **答：** 如果使用完全无标注的数据作为未标注集，可以在这些无标注数据的标注信息中添加任意一个定位框，该定位框的标注格式需要和其他有标注数据的标注格式保持一致。
    之后，在未标注集的 txt 索引中加入该数据的文件名即可。

4.  **问： 当将骨干网络从 RetinaNet 改为自定义的 SSD 之后报错：`TypeError: init() missing 1 required positional argument: 'input_size'`。（问题 [#30](../../../issues/30)）**

    **答：** 请在自定义配置文件 `ssd300.py` 中的 dict `model.backbone` 中添加 `input_size=input_size`。为避免更多潜在问题，请在 [2.3.0 版本](https://github.com/open-mmlab/mmdetection/tree/v2.3.0/) 而不是 [最新版本](https://github.com/open-mmlab/mmdetection/) 中自定义修改 MMDetection 内的任何文件。

5.  **问： 当使用自己的数据集（只有行人类）测试代码时，`l_det_loc`、`L_det` 损失是 nan。（问题 [#34](../../../issues/34) 和 [#35](../../../issues/35)）**

    **答：** 请检查数据集的边界框标注信息是否有问题。
