# MS COCO 的数据集准备

语言：简体中文 | [English](README.md)

该目录下有 5 个 python 文件，它们被用来把 MS COCO 数据集的 JSON 格式标注转换为 PASCAL VOC 数据集的 XML 格式标注（参考自 [KapilM26/coco2VOC](https://github.com/KapilM26/coco2VOC)）。

以下是 MS COCO 数据集准备的说明。请确保你已经完成了 [安装文档](../docs/installation_cn.md) 中的安装过程。

1. 请如下所示新建一个文件目录树，其中 MS COCO 数据集的图像与标注（ *train* 部分和 *val* 部分）可以从此处下载：

    图像（ *train* 部分）：http://images.cocodataset.org/zips/train2017.zip
    
    图像（ *val* 部分）：http://images.cocodataset.org/zips/val2017.zip
    
    标注：http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    ```
    ├── coco（从官网下载）
    │   ├── annotations
    │   │   ├── instances_train2017.json
    │   │   ├── instances_val2017.json
    │   ├── images（train2017 和 val2017 的组合）
    │   ├── train2017
    │   ├── val2017
    ├── coco2voc（手动新建）
    │   ├── Annotations（空目录）
    │   ├── ImageSets
    │   │   ├── Main（空目录）
    │   ├── JPEGImages（指向 coco/images 的软链接）
    ```
    
    你也可以直接使用下面的命令行：
    
    ```bash
    cd $你的数据集地址
    mkdir coco
    cd coco
    wget http://images.cocodataset.org/zips/train2017.zip  # 此步比较耗时
    unzip train2017.zip  # 此步比较耗时
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    mkdir images
    cp train2017/* images/
    cp val2017/* images/
    cd ../
    mkdir coco2voc
    mkdir cocovoc/Annotations
    mkdir coco2voc/ImageSets
    mkdir coco2voc/ImageSets/Main
    ln -s coco/images coco2voc/JPEGImages
    ```
    
2. 请修改这个目录下 `MIAOD.py` 中对应的数据集地址部分，它们位于：

    ```python
    第 1 行：data_root_coco='$YOUR_DATASET_PATH/coco/'
    第 2 行：data_root_voc='$YOUR_DATASET_PATH/coco2voc/'
    ```
    
    请把上面的 `$YOUR_DATASET_PATH` 和 `$你的数据集地址` 改为你实际的数据集地址（即你新建 `coco` 文件夹的地址）。
    
    地址请使用绝对路径（如以 `/` 开始的路径），不要使用相对路径（如以 `./` 或 `../` 开始的路径）。
    
3. 请将这4个文件复制到如下地址，并替换原有文件：

    - active_datasets.py -> mmdet/utils/active_datasets.py
    - MIAOD.py -> configs/MIAOD.py
    - train.py -> tools/train.py
    - voc.py -> mmdet/datasets/voc.py
   
   命令行如下：
   
   ```bash
   cp active_datasets.py ../mmdet/utils/active_datasets.py
   cp MIAOD.py ../configs/MIAOD.py
   cp train.py ../tools/train.py
   cp voc.py ../mmdet/datasets/voc.py
   ```

4. 请按照如下命令行安装 `pascal_voc_writer` 包以写入 PASCAL VOC 数据集的 XML 标注。

    ```bash
    pip install pascal_voc_writer
    ```
    
5. 请按照如下命令行将 MS COCO 数据集的 JSON 标注格式转换为 PASCAL VOC 数据集的 XML 标注格式。

    ```bash
    python coco2voc.py \
        --ann_file $你的数据集地址/coco/annotations/instances_train2017.json \
        --output_dir $你的数据集地址/coco2voc/Annotations/
    ```
    
   请把上面的 `$你的数据集地址` 改为你实际的数据集地址（即你新建 `coco` 文件夹的地址）。

现在你已经完成了 MS COCO 的数据集准备。请按照 [这里](../README_cn.md#训练和测试) 来完成剩余的训练与测试步骤，并且在 MS COCO 数据集的测试步骤中把 `--eval mAP` 替换为 `--eval bbox`。
