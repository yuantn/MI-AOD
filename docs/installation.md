# Installation

Language: [简体中文](installation_cn.md) | English

<!-- TOC -->

- [Prerequisites](#prerequisites)
- [Environment Installation](#environment-installation)
- [Modification in the MMCV Package](#modification-in-the-mmcv-package)
- [A From-scratch Setup Script](#a-from-scratch-setup-script)

<!-- TOC -->

## Prerequisites

- GPU
- Linux platform (Ubuntu and CentOS are recommended, **Ubuntu 16.04, Ubuntu 18.04** and **CentOS 7.6** have been tested.)
- [Anaconda3](https://www.anaconda.com/)
- Python 3.6+ (**Python 3.7** is recommended and has been tested.)
- [PyTorch 1.3+](https://pytorch.org/) (**PyTorch 1.6** is recommended and has been tested.)
- [CUDA 9.2+](https://developer.nvidia.com/cuda-toolkit-archive) (**CUDA 10.2** is recommended and has been tested. If you build PyTorch from source, CUDA 9.0 is also compatible.)
- [CuDNN](https://developer.nvidia.com/cudnn) (Optional. **CuDNN 7.6.5** has been tested.)
- GCC 5+ (**GCC 4.8.5, 5.5.0 and 7.5.0** are recommended and have been tested.)
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) (**MMCV 1.0.5** is highly recommended and have been tested. It is the only MMCV version compatible with [MMDetection 2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0), which is the base of this MI-AOD repository.)

[Here](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md#prerequisites) are the compatible MMDetection and MMCV versions.
Please install the correct version of MMCV to avoid installation issues.

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Environment Installation

Before everything starts, please ensure that you already have **Anaconda3** and **CUDA 10.2** installed.

<!-- 0. You can simply install mmdetection with the following commands:
    `pip install mmdet` -->

1. Please create a conda virtual environment with **Python 3.7** and activate it.

    ```shell
    conda create -n miaod python=3.7 -y
    conda activate miaod
    ```

2. Please install **PyTorch 1.6.0** and **torchvision 0.7.0** for **CUDA 10.2** following the [official instructions](https://pytorch.org/get-started/previous-versions/#v160), e.g.,

    ```shell
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
    ```

    Please note: make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/get-started/previous-versions/#v160).

    If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

3. Please install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{mmcv_version}`, `{cu_version}` and `{torch_version}` in the url to your desired one. **Our recommended is, to install the `mmcv-full 1.0.5` with `CUDA 10.2` and `PyTorch 1.6.0`**, use the following command:

    ```shell
    pip install mmcv-full==1.0.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
    ```

    Please see [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.0.5.zip
    unzip mmcv-1.0.5
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full==1.0.5
    ```

4. Please clone the MI-AOD repository.

    ```shell
    git clone https://github.com/yuantn/MI-AOD.git
    cd MI-AOD
    ```

    If it is too slow, you can also try downloading the repository like this:

    ```shell
    wget https://github.com/yuantn/MI-AOD/archive/master.zip
    unzip master.zip
    cd MI-AOD-master
    ```

5. Please install build requirements and then install and compile MMDetection in MI-AOD.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Please note:

a. Following the above instructions, MMDetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`, you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements.
To use optional dependencies like `albumentations` and `imagecorruptions`, either install them manually with `pip install -r requirements/optional.txt`,
or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

## Modification in the MMCV Package

To train with two dataloaders (i.e., the labeled set dataloader and the unlabeled set dataloader mentioned in the paper) at the same time,
you will need to modify the `epoch_based_runner.py` in the mmcv package.

Considering that this will affect all code that uses this environment,
so we suggest you set up a separate environment for MI-AOD (i.e., the `miaod` environment created above).

```shell
cp -v epoch_based_runner.py $YOUR_ANACONDA_PATH/envs/miaod/lib/python3.7/site-packages/mmcv/runner/
```

Please change the `$YOUR_ANACONDA_PATH` to your actual Anaconda3 installation directory. Usually it would be `~/anaconda3`.

After that, if you have modified anything in the mmcv package
(including but not limited to: updating/re-installing Python, PyTorch, mmdetection, mmcv, mmcv-full, conda environment),
you are supposed to copy the `epoch_base_runner.py` provided in this repository to the mmcv directory again. ([Issue #3](../../../issues/3))

## A From-scratch Setup Script

Assuming that you have **Anaconda3** and **CUDA 10.2** installed already, here is a full script for setting up MI-AOD with conda.

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

If you have any question, please feel free to leave an issue [here](../../../issues), or refer to [install.md in MMDetection V2.3.0](https://github.com/open-mmlab/mmdetection/blob/v2.3.0/docs/install.md).

And please refer to [FAQ](FAQ.md) for frequently asked questions.
