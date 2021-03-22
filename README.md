# MI-AOD

Language: [简体中文](./README_cn.md) | English

## Introduction

[comment]: <This is the code for [*Multiple Instance Active Learning for Object Detection*](https://github.com/yuantn/MI-AOD/raw/master/paper.pdf), CVPR 2021.>
This is the code for *Multiple Instance Active Learning for Object Detection* (The PDF is not available temporarily), CVPR 2021.

[comment]: <In this paper, we propose Multiple Instance Active Object Detection (MI-AOD), to select the most informative images for detector training by observing instance-level uncertainty. MI-AOD defines an instance uncertainty learning module, which leverages the discrepancy of two adversarial instance classifiers trained on the labeled set to predict instance uncertainty of the unlabeled set. MI-AOD treats unlabeled images as instance bags and feature anchors in images as instances, and estimates the image uncertainty by re-weighting instances in a multiple instance learning (MIL) fashion. Iterative instance uncertainty learning and re-weighting facilitate suppressing noisy instances, toward bridging the gap between instance uncertainty and image-level uncertainty. ![Illustration](./CVPR-MI-AOD.png) Experiments validate that MI-AOD sets a solid baseline for instance-level active learning. On commonly used object detection datasets, MI-AOD outperforms state-of-the-art methods with significant margins, particularly when the labeled sets are small. ![Results](./Results.png)>
Other introduction and figures are not available temporarily.

## Installation

A Linux platform (Ours are Ubuntu 18.04 LTS) and [anaconda3](https://www.anaconda.com/) is recommended, since they can install and manage environments and packages conveniently and efficiently.

A TITAN V GPU and [CUDA 10.2](https://developer.nvidia.com/cuda-toolkit-archive) with [CuDNN 7.6.5](https://developer.nvidia.com/cudnn) is recommended, since they can speed up model training.

After anaconda3 installation, you can create a conda environment as below:

```
conda create -n miaod python=3.7 -y
conda activate miaod
```

Please refer to [MMDetection v2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) and the [install.md](https://github.com/open-mmlab/mmdetection/blob/v2.3.0/docs/install.md) of it for environment installation.

And then please clone this repository as below:

```
git clone https://github.com/yuantn/MI-AOD.git
cd MI-AOD
```

If it is too slow, you can also try downloading the repository like this:

```
wget https://github.com/yuantn/MI-AOD/archive/master.zip
unzip MI-AOD.zip
cd MI-AOD-master
```

## Modification in the mmcv Package

To train with two dataloaders (i.e., the labeled set dataloader and the unlabeled set dataloader mentioned in the paper) at the same time, you will need to modify the ` epoch_based_runner.py ` in the mmcv package.

Considering that this will affect all code that uses this environment, so we suggest you set up a separate environment for MI-AOD (i.e., the ` miaod ` environment created above).

```
cp -v epoch_based_runner.py ~/anaconda3/envs/miaod/lib/python3.7/site-packages/mmcv/runner/
```

After that, if you have modified anything in the mmcv package (including but not limited to: updating/re-installing Python, PyTorch, mmdetection, mmcv, mmcv-full, conda environment), you are supposed to copy the “epoch_base_runner.py” provided in this repository to the mmcv directory again. ([Issue #3](../../issues/3))

## Datasets Preparation

Please download VOC2007 datasets ( *trainval* + *test* ) and VOC2012 datasets ( *trainval* ) from:

VOC2007 ( *trainval* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

VOC2007 ( *test* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

VOC2012 ( *trainval* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

And after that, please ensure the file directory tree is as below:
```
├── VOCdevkit
│   ├── VOC2007
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   ├── VOC2012
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
```
You may also use the following commands directly:
```
cd $YOUR_DATASET_PATH
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
```
After that, please modify the corresponding dataset directory in this repository, they are located in:
```
Line 1 of configs/MIAL.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
Line 1 of configs/_base_/voc0712.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
```
Please change the ` $YOUR_DATASET_PATH `s above to your actual dataset directory (i.e., the directory where you intend to put the downloaded VOC tar file).

And please use the absolute path (i.e., start with ` / `) but not a relative path (i.e., start with ` ./ ` or ` ../ `）.

## Training and Test

We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

And we also recommend you to use a single GPU, because the usage of multi-GPU may result in errors caused by the multi-processing of the dataloader.

If you use only a single GPU, you can use the ` script.sh ` file directly as below:
```
chmod 777 ./script.sh
./script.sh $YOUR_GPU_ID
```
Please change the ` $YOUR_GPU_ID ` above to your actual GPU ID number (usually a non-negative number).

Please ignore the error ` rm: cannot remove './log_nohup/nohup_$YOUR_GPU_ID.log': No such file or directory ` if you run the ` script.sh ` file for the first time.

The ` script.sh ` file will use the GPU with the ID number ` $YOUR_GPU_ID ` and PORT `(30000+$YOUR_GPU_ID*100)` to train and test.

The log file will not flush in the terminal, but will be saved and updated in the file `./log_nohup/nohup_$YOUR_GPU_ID.log` and ` ./work_dirs/MIAL/$TIMESTAMP.log ` . These two logs are the same. You can change the directories and names of the latter log files in Line 48 of `./configs/MIAL.py` .

You can also use other files in the directory ` './work_dirs/MIAL/ ` if you like, they are as follows:

- **JSON file `$TIMESTAMP.log.json`**

  You can load the losses and mAPs during training and test from it more conveniently than from the `./work_dirs/MIAL/$TIMESTAMP.log` file.

- **npy file `X_L_$CYCLE.npy` and `X_U_$CYCLE.npy`**

  The `$CYCLE` is an integer from 0 to 6, which are the active learning cycles.

  You can load the indexes of the labeled set and unlabeled set for each cycle from them.

  The indexes are the integers from 0 to 16550 for PASCAL VOC datasets, where 0 to 5010 is for PASCAL VOC 2007 *trainval* set and 5011 to 16550 for PASCAL VOC 2012 *trainval* set.

  An example code for loading these files is the Line 108-114 in the `./tools/train.py` file (which are in comments now).

- **pth file `epoch_$EPOCH.pth` and `latest.pth`**

  The `$EPOCH` is an integer from 0 to 2, which are the epochs of the last label set training.

  You can load the model state dictionary from them.

  An example code for loading these files is the Line 109, 143-145 in the `./tools/train.py` file (which are in comments now).

- **txt file `trainval_L_07.txt`, `trainval_U_07.txt`, `trainval_L_12.txt` and `trainval_U_12.txt` in each `cycle$CYCLE` directory**

  The `$CYCLE` is the same as above.

  You can load the names of JPEG images of the labeled set and unlabeled set for each cycle from them.

  "L" is for the labeled set and "U" is for the unlabeled set. "07" is for the PASCAL VOC 2007 *trainval* set and "12" is for the PASCAL VOC 2012 *trainval* set.
  
An example output folder is provided on Google Drive and Baidu Drive, including the log file, the last trained model, and all other files above.

- **Google Drive:**

  [Log file](https://drive.google.com/file/d/1dC2k3SCC_C9yvp2oIlStiVHbsyyh2QuC/view?usp=sharing)
  
  [Last trained model (latest.pth)](https://drive.google.com/file/d/1gOaN3_R_QmeJ2bz0hczDmOXERTvMeSut/view?usp=sharing)
  
  [The whole example output folder](https://drive.google.com/file/d/1oRiT-BBx8wlTWaXEO1_3xSGls9YeiCDA/view?usp=sharing)

- **Baidu Drive:**

  [Log file (Extraction code: kqsj)](https://pan.baidu.com/s/1FL7si7fxX86vwqqaYC3B_g)
  
  [Last trained model (latest.pth) (Extraction code: 80v5)](https://pan.baidu.com/s/1EV4V-N1TeLc8IAF5rC0y2A)
  
  [The whole example output folder (Extraction code: 6kn2)](https://pan.baidu.com/s/1v_4frByp1_dNiPA_cuMqwQ)

## Code Structure
```
├── $YOUR_ANACONDA_DIRECTORY
│   ├── anaconda3
│   │   ├── envs
│   │   │   ├── miaod
│   │   │   │   ├── lib
│   │   │   │   │   ├── python3.7
│   │   │   │   │   │   ├── site-packages
│   │   │   │   │   │   │   ├── mmcv
│   │   │   │   │   │   │   │   ├── runner
│   │   │   │   │   │   │   │   │   ├── epoch_based_runner.py
│
├── ...
│
├── configs
│   ├── _base_
│   │   ├── default_runtime.py
│   │   ├── retinanet_r50_fpn.py
│   │   ├── voc0712.py
│   ├── MIAL.py
│── log_nohup
├── mmdet
│   ├── apis
│   │   ├── __init__.py
│   │   ├── test.py
│   │   ├── train.py
│   ├── models
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── MIAL_head.py
│   │   │   ├── MIAL_retina_head.py
│   │   │   ├── base_dense_head.py 
│   │   ├── detectors
│   │   │   ├── base.py
│   │   │   ├── single_stage.py
│   ├── utils
│   │   ├── active_datasets.py
├── tools
│   ├── train.py
├── work_dirs
│   ├── MIAL
├── script.sh
```

The code files and folders shown above are the main part of MI-AOD, while other code files and folders are created following MMDetection to avoid potential problems.

The explanation of each code file or folder is as follows:

- **epoch_based_runner.py**: Code for training and test in each epoch, which can be called by `./apis/train.py`.

- **configs**: Configuration folder, including running settings, model settings, dataset settings and other custom settings for active learning and MI-AOD.

  - **\_\_base\_\_**: Base configuration folder provided by MMDetection, which only need a little modification and then can be recalled by `.configs/MIAL.py`.

    - **default_runtime.py**: Configuration code for running settings, which can be called by `./configs/MIAL.py`.
  
    - **retinanet_r50_fpn.py**: Configuration code for model training and test settings, which can be called by `./configs/MIAL.py`.
 
    - **voc0712.py**: Configuration code for PASCAL VOC dataset settings and data preprocessing, which can be called by `./configs/MIAL.py`.
  
  - **MIAL.py**: Configuration code in general including most custom settings, containing active learning dataset settings, model training and test parameter settings, custom hyper-parameter settings, log file and model saving settings, which can be mainly called by `./tools/train.py`. The more detailed introduction of each parameter is in the comments of this file.

- **log_nohup**: Log folder for storing log output on each GPU temporarily.

- **mmdet**: The core code folder for MI-AOD, including intermidiate training code, object detectors and detection heads and active learning dataset establishment.

  - **apis**: The inner training, test and calculating uncertainty code folder of MI-AOD.
  
    - **\_\_init\_\_.py**: Some function initialization in the current folder.
    
    - **test.py**: Code for testing the model and calculating uncertainty, which can be called by `epoch_based_runner.py` and `./tools/train.py`.
    
    - **train.py**: Code for setting random seed and creating training dataloaders to prepare for the following epoch-level training, which can be called by `./tools/train.py`.
    
  - **models**: The code folder with the details of network model architecture, training loss, forward propagation in test and calculating uncertainty.
  
    - **dense_heads**: The code folder of training loss and the network model architecture, especially the well-designed head architecture.
    
      - **\_\_init\_\_.py**: Some function initialization in the current folder.
      
      - **MIAL_head.py**: Code for forwarding anchor-level model output, calculating anchor-level loss, generating pseudo labels and getting bounding boxes from existing model output in more details, which can be called by `./mmdet/models/dense_heads/base_dense_head.py` and `./mmdet/models/detectors/single_stage.py`.
      
      - **MIAL_retina_head.py**: Code for building the MI-AOD model architecture, especially the well-designed head architecture, and define the forward output, which can be called by `./mmdet/models/dense_heads/MIAL_head.py`.
      
      - **base_dense_head.py**: Code for choosing different equations to calculate loss, which can be called by `./mmdet/models/detectors/single_stage.py`.
      
    - **detectors**: The code folder of the forward propogation and backward propogation in the overall training, test and calculating uncertainty process.
    
      - **base.py**: Code for arranging training loss to print and returning the loss and image information, which can be called by `epoch_based_runner.py`.

      - **single_stage.py**: Code for extracting image features, getting bounding boxes from the model output and returning the loss, which can be called by `./mmdet/models/detectors/base.py`.
      
  - **utils**: The code folder for creating active learning datasets.

    - **active_dataset.py**: Code for creating active learning datasets, including creating initial labeled set, creating the image name file for the labeled set and unlabeled set and updating the labeled set after each active learning cycle, which can be called by `./tools/train.py`.

- **tools**: The outer training and test code folder of MI-AOD.

  - **train.py**: Outer code for training and test for MI-AOD, including generating PASCAL VOC datasets for active learning, loading image sets and models, Instance Uncertainty Re-weighting and Informative Image Selection in general, which can be called by `./script.sh`.

- **work_dirs**: Work directory of the index and image name of the labeled set and unlabeled set for each cycle, all log and json outputs and the model state dictionary for the last 3 cycle, which are introduced in the **Training and Test** part above.

- **script.sh**: The script to run MI-AOD on a single GPU. You can run it to train and test MI-AOD simply and directly mentioned in the **Training and Test** part above as long as you have prepared the conda environment and PASCAL VOC 2007+2012 datasets.

## Citation

[comment]: <If you find this repository useful for your publications, please consider citing our [paper](https://github.com/yuantn/MI-AOD/raw/master/paper.pdf).>
If you find this repository useful for your publications, please consider citing our paper. (The PDF is not available temporarily)
```angular2html
@inproceedings{MIAOD2021,
    author    = {Tianning Yuan and
                 Fang Wan and
                 Mengying Fu and
                 Jianzhuang Liu and
                 Songcen Xu and
                 Xiangyang Ji and
                 Qixiang Ye},
    title     = {Multiple Instance Active Learning for Object Detection},
    booktitle = {CVPR},
    year      = {2021}
}
```

## Acknowledgement

In this repository, we reimplemented RetinaNet on PyTorch based on [mmdetection](https://github.com/open-mmlab/mmdetection).
