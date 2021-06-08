# 安装

语言： 简体中文 | [English](installation.md)

<!-- TOC -->

- [安装前准备](#安装前准备)
- [环境安装](#环境安装)
- [修改 MMCV 包](#修改-mmcv-包)
- [从头开始的安装脚本](#从头开始的安装脚本)

<!-- TOC -->

## 安装前准备

- GPU
- Linux 开发平台 （推荐使用 Ubuntu 和 CentOS，我们在 **Ubuntu 16.04**、**Ubuntu 18.04**、**CentOS 7.6** 上测试过。）
- [Anaconda3](https://www.anaconda.com/)
- Python 3.6+ （推荐使用 **Python 3.7**，我们用它测试过。）
- [PyTorch 1.3+](https://pytorch.org/) （推荐使用 **PyTorch 1.6**，我们用它测试过。）
- [CUDA 9.2+](https://developer.nvidia.com/cuda-toolkit-archive) （推荐使用 **CUDA 10.2**，我们用它测试过。如果你从源代码来搭建 PyTorch，那么 CUDA 9.0 也可以兼容。）
- [CuDNN](https://developer.nvidia.com/cudnn) （可选项，我们用 **CuDNN 7.6.5** 测试过。）
- GCC 5+ （推荐使用 **GCC 4.8.5, 5.5.0 and 7.5.0**，我们用它们测试过。）
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) （非常推荐使用 **MMCV 1.0.5**，我们用它测试过。它是唯一一个和 [MMDetection 2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) 兼容的 MMCV 版本，这套 MI-AOD 代码是在该版本的 MMDetection 基础上编写的。

[这里](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md#prerequisites) 是可兼容的 MMDetection 和 MMCV 版本。
为避免安装上的问题，请安装正确的 MMCV 版本。

注意：如果你安装了 mmcv，你需要先运行 `pip uninstall mmcv`。
如果安装了 mmcv 和 mmcv-full，那将会报错 `ModuleNotFoundError`。

## 环境安装

在一切开始前，请确保你已经安装了 **Anaconda3** 和 **CUDA 10.2**。

<!-- 0. 你可以直接用如下命令简单安装 mmdetection：
    `pip install mmdet` -->

1. 请用 **Python 3.7** 创建一个 conda 虚拟环境，并激活它。

    ```shell
    conda create -n miaod python=3.7 -y
    conda activate miaod
    ```

2. 请根据 [官方说明](https://pytorch.org/get-started/previous-versions/#v160) 安装与 **CUDA 10.2** 匹配的 **PyTorch 1.6.0** 和 **torchvision 0.7.0**，例如：

    ```shell
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
    ```

    请注意：确保你编译的 CUDA 版本和运行的 CUDA 版本相匹配。
    你可以在 [Pytorch 网站](https://pytorch.org/get-started/previous-versions/#v160) 上检查支持的 CUDA 版本的预编译包。
    
    如果你从源代码搭建 PyTorch，而不是安装预编译包的话，你可以用更多版本的 CUDA （如 9.0）。

3. 请安装 mmcv-full，我们推荐你用如下的命令安装搭建好的包。

    ```shell
    pip install mmcv-full=={mmcv_版本} -f https://download.openmmlab.com/mmcv/dist/{cuda_版本}/{torch_版本}/index.html
    ```

    请将网址中的 `{mmcv_版本}`、`{cuda_版本}`、`{torch_版本}` 替换为你想要的版本。**我们推荐，安装 `CUDA 10.2` 和 `PyTorch 1.6.0` 下的 `mmcv-full 1.0.5`**，命令行如下：

    ```shell
    pip install mmcv-full==1.0.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
    ```

    关于与不同 PyTorch 和 CUDA 版本兼容的 MMCV 版本，请参见 [这里](https://github.com/open-mmlab/mmcv#installation)。
    或者你也可以选择用如下命令行从源代码编译 mmcv：
    
    ```shell
    wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.0.5.zip
    unzip mmcv-1.0.5
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    或者直接运行
  
    ```shell
    pip install mmcv-full==1.0.5
    ```
  
4. 请克隆 MI-AOD 代码库。

    ```shell
    git clone https://github.com/yuantn/MI-AOD.git
    cd MI-AOD
    ```
    
    如果那样太慢的话，你也可以尝试像这样直接下载这个代码库：

    ```shell
    wget https://github.com/yuantn/MI-AOD/archive/master.zip
    unzip master.zip
    cd MI-AOD-master
    ```

5. 请安装搭建代码需要的包，然后在 MI-AOD 中安装与编译 MMDetection。

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```
    
请注意：

a. 按照上面的说明，MMDetection 是以 `dev` 模式安装的，任何本地对代码的修改都会生效，而不用重新安装它。

b. 如果你愿意使用 `opencv-python-headless` 而不是 `opencv-python`，你可以在安装 MMCV 之前安装它。

c. 一些依赖项是可选的。若仅以最小运行需求安装，直接运行 `pip install -v -e .` 即可。
若要使用像 `albumentations` 和 `imagecorruptions` 之类的可选依赖项，则可手动使用 `pip install -r requirements/optional.txt` 安装它们，
也可以在调用 `pip` 命令时明确额外的需求 （例如，`pip install -v -e .[optional]`）。
对于额外的域而言，有效键为：`all`、`tests`、`build`、`optional`。

## 修改 MMCV 包

为了能够同时训练两个 dataloader（即论文中提到的有标号的 dataloader 和无标号的 dataloader），需要修改 mmcv 包中的 `epoch_based_runner.py` 文件。

考虑到这会影响到所有使用这个环境的代码，
所以我们建议为 MI-AOD 创建一个单独的环境（即上文中创建的 `miaod` 环境）。

```shell
cp -v epoch_based_runner.py $你的_ANACONDA_安装地址/envs/miaod/lib/python3.7/site-packages/mmcv/runner/
```

请将 `$你的_ANACONDA_安装地址` 改为你实际的 Anaconda3 安装目录。通常它是 `~/anaconda3`。

之后如果你修改了 mmcv 包中的任何文件
（包括但不限于：更新/重新安装了 Python、PyTorch、mmdetection、mmcv、mmcv-full、conda 环境），
都应该重新将这个代码库中的 `epoch_base_runner.py` 文件再次复制到上面的 mmcv 文件夹下。([Issue #3](../../issues/3))

## 从头开始的安装脚本

假设你已经安装好了 **Anaconda3** 和 **CUDA 10.2**，如下为一个用 conda 设置 MI-AOD 的完整脚本。

```shell
conda create -n miaod python=3.7 -y
conda activate miaod

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

# install the mmcv==1.0.5
pip install mmcv-full==1.0.5+torch1.6.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html

# install mmdetection
git clone https://github.com/yuantn/MI-AOD.git
cd MI-AOD
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

# modify epoch_based_runner.py
cp -v epoch_based_runner.py $YOUR_ANACONDA_PATH/envs/miaod/lib/python3.7/site-packages/mmcv/runner/
```

如果你有任何问题，请随时在 [问题](../../issues) 中留言，或者参考 [MMDetection 2.3.0 版本的安装说明](https://github.com/open-mmlab/mmdetection/blob/v2.3.0/docs/install.md)。

请参考 [常见问题解答](FAQ_cn.md) 来查看大家的常见问题。
