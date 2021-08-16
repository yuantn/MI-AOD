# Frequently Asked Questions (FAQ)

Language: [简体中文](FAQ_cn.md) | English

We have list some common troubles faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.
If the contents here do not cover your issue, please just create an issue [here](../../../issues).
The open issues are not included here for now, just in case someone will ask further questions.

<!-- TOC -->

- [Environment Installation](#environment-installation)
- [Training and Test](#training-and-test)
- [Paper Details](#paper-details)
- [Fixed Bugs and New Features](#fixed-bugs-and-new-features)

<!-- TOC -->

## Environment Installation

1.  **Q: `TypeError: forward() missing 1 required positional argument: 'x'`. (Issues [#3](../../../issues/3), [#5](../../../issues/5) and [#15](../../../issues/15#issuecomment-854458413))**
    
    **A:** Please refer to [Modification in the mmcv Package](installation.md#modification-in-the-mmcv-package).
    That is, you are supposed to copy the `epoch_based_runner.py` provided in this repository to the mmcv directory again (as described in the installation.md)
    if you have modified anything in the mmcv package (including but not limited to: updating/re-installing Python, PyTorch, mmdetection, mmcv, mmcv-full, conda environment).

2.  **Q: `AssertionError: MMCV==1.3.1 is used but incompatible. Please install mmcv>=1.0.5, <=1.0.5`. (Issue [#10](../../../issues/10))**

    **A:** Please uninstall **mmcv** and **mmcv-full**, and then reinstall **mmcv-full==1.0.5**.
    
3.  **Q: After installing mmcv==1.0.5, there are still some errors:**
    
    ```bash
    ImportError: cannot import name 'Config' from 'mmcv' (unknown location)
    ModuleNotFoundError: No module named 'mmcv.utils'
    ```
    
    **(Issue [#13](../../../issues/13#issuecomment-841080219))**
    
    **A:** Please refer to the step 5 [here](../../../blob/master/docs/installation.md#environment-installation) to install build requirements and install and compile MMDetection.
    

## Training and Test

1.  **Q: `AttributeError: 'Tensor' object has no attribute 'isnan'`. (Issues [#2](../../../issues/2) and [#9](../../../issues/9))**

    **A:** Option 1. Re-install the **Pytorch==1.6.0** and **TorchVision==0.7.0** with the [PyTorch official instructions](https://pytorch.org/get-started/previous-versions/#v160).
    
    Option 2. Check the lines of `AttributeError`, and replace the `if value.isnan()` with `if value != value` ( considering that only nan != nan).
    
    The error must be in the Line 483 and 569 of the `./mmdet/models/dense_heads/MIAOD_head.py`.
    
2.  **Q: There is not any reaction when running `./script.sh 0`. (Issues [#6](../../../issues/6) and [#13](../../../issues/13))**

    **A:** When running `script.sh`, the code is executed in the background.
    You can view the output log by running this command in the root directory: `vim log_nohup/nohup_0.log`.
    
    There is another solution to flush the logs in the terminal [in another section](#fixed-bugs-and-new-features).
    
3.  **Q: `StopIteration`. (Issues [#7](../../../issues/7#issuecomment-823068004) and [#11](../../../issues/11))**

    **A:** Thanks for the solution from [@KevinChow](https://github.com/kevinchow1993).
    
    In the functions `create_X_L_file()` and `create_X_U_file()` of `mmdet/utils/active_datasets.py`, before writing into the txt files,
    sleep for some time randomly to make them write files not at the same time:

    ```python
        time.sleep(random.uniform(0,3))
        if not osp.exists(save_path):
            mmcv.mkdir_or_exist(save_folder)
            np.savetxt(save_path, ann[X_L_single], fmt='%s')
    ```

    After calling `create_X_L_file()` and `create_X_U_file()` in the `tools/train.py`, sychronize the threads on each GPUs by adding:

    ```python
              if dist.is_initialized():
                  torch.distributed.barrier()
    ```

4.  **Q: I want to run MI-AOD with other data, which files should I modify? (Issue [#13](../../../issues/13#issuecomment-845709365))**

    **A:** You should only modify `configs/MIAOD.py` if you can convert your other training and test data into PASCAL VOC format. It contains all parameters and settings.
    
5.  **Q: Validation error: `TypeError: 'DataContainer' object is not subscriptable`. (Issue [#14](../../../issues/14))**

    **A:** In `get_bboxes` function of `mmdet/models/dense_heads/MIAOD_head.py`, please change
    
    ```python
    img_shape = img_metas[img_id]['img_shape']
    ```
    
    to
    
    ```python
    img_shape = img_metas.data[0]
    ```
    
    Note: You only need to make changes when you encounter this problem, usually it won't occur on a GPU environment.

6.  **Q: What is `$CONFIG_PATH` and `$CKPT_PATH` in `python tools/test.py $CONFIG_PATH $CKPT_PATH`? (Issue [#17](../../../issues/17))**

    **A:** Please refer to [here](../../../#train-and-test) for explanation. That is:
    
    > where $CONFIG_PATH should be replaced by the path of the config file in the configs folder (usually it would be configs/MIAOD.py)
    
    > $CKPT_PATH should be replaced by the path of the checkpoint file (*.pth) in the work_dirs folder after training.

7.  **Q: When training on custom dataset (only 1 foreground class), why is l_imgcls always 0 during training? (Issues [#23](../../../issues/23) and [#24](../../../issues/24))**

    **A:** To avoid that, you can create another class without any corresponding image in the dataset.
    
8.  **Q: In `tools/train.py`, is it first trained on the labeled dataset? What is the purpose? (Issue [#25](../../../issues/26))**

    **A:** It is necessary to train on the labeled set for the first and last epochs to ensure the stability of the training model.
    
9.  **Q: For the unlabeled set, why are the operations on GT information (ie, `gt_bboxes` and `gt_labels`) also involved in lines 70-74 of `epoch_based_runner.py`? If the completely unlabeled data is used as the unlabeled set, what needs to be modified? (Issues [#28](../../../issues/28) and [#29](../../../issues/29))**

    **A:** These lines are to remove the localization information of the images in the unlabeled set.
    In this way, when calculating the loss on the unlabeled set, we can know the data source without backward propagating the gradient.
    In fact, the GT information has not been used.
    
    If the completely unlabeled data is used as the unlabeled set, you can add any bounding box to the annotation of the unlabeled data arbitrarily.
    The annotation format of the bounding box needs to be consistent with that of other labeled data.
    After that, just add the file name to the txt index of the unlabeled data set.


## Paper Details

1.  **Q: Will the code be open sourced to MMDetection for wider spread? (Issue [#1](../../../issues/1))**

    **A:** MI-AOD is mainly for active learning, but MMDetection is more for object detection.
    It would be better for MI-AOD to open source to an active learning toolbox. 

2.  **Q: There are differences on the order of maximizing/minimizing uncertainty and the fixed layers between paper and code. (Issues [#4](../../../issues/4) and [#16](../../../issues/16#issuecomment-859363894))**

    **A:** Our experiments have shown that, if the order of max step and min step is reversed (including the fixed layers), the performance will change little.
        
3.  **Q: The initial labeled experiment in Figure 5 of this paper should be similar in theory. Why not in experiments? (Issue [#4](../../../issues/4#issuecomment-800871469))**

    **A:** The reason can be summarized as:
    - Intentional use of unlabeled data
    - -> Better aligned instance distributions of the labeled and unlabeled set
    - -> Effective information (prediction discrepancy) of the unlabeled set
    - -> Naturally formed unsupervised learning procedure
    - -> Performance improvement

4.  **Q: How to guarantee the distribution bias between the labeled data and the unlabeled data is minimized based on my derivation? (Issue [#8](../../../issues/8))**

    **A:** There is something wrong in the process and result of your derivation.
    And minimizing the distribution bias is achieved by two steps (maximizing and minimizing uncertainty, as shown in Fig. 2(a) ) but not only minimizing uncertainty.

5.  **Q: What is the main difference between the active learning and semi-supervision, and can I directly use active learning for semi-supervision? (Issue [#12](../../../issues/12))**

    **A:** The core of active learning is that we first train a model with small amount of data,
    and then calculate the uncertainty (or other designed metrics) to select the informative samples for the next active learning cycle.
    However, the semi-supervised learning tries to mine and utilize unlabeled samples in a static perspective but not a dynamic perspective.
    
    I think that our work MI-AOD cleverly combine the active learning with semi-supervised learning.
    That is, we use semi-supervised learning (or its key idea) to learn with limited labeled data and enough unlabeled data, 
    and use active learning to select informative unlabeled data and annotate them.
    This is the trend of the recent research in active learning, and use active learning for semi-supervised learning is also a good idea.
    
6.  **Q: There are differences on the `y_head_cls` (in Eq. (5) of the paper, and `forward_single` function in `mmdet/dense_heads/MIAOD_retina_head.py` of the code) between paper and code. What does the `maximum` and `softmax` function in the code mean? (Issue [#16](../../../issues/16))**

    **A:** The equation in the code is:
    
    ```python
    y_head_cls = y_head_f_mil.softmax(2) * y_head_cls_term2.sigmoid().max(2, keepdim=True)[0].softmax(1)
    ```
    
    which can be simplified to:
    
    ```python
    y_head_cls = A.softmax() * B.max().softmax()
    ```
    
    where A and B are the output of MIL head and averaged classifier heads.
    
    `max(2, keepdim=True)[0]` is to highlight the class with the highest score, which are most likely to be predicted as the foreground.
    
    `softmax(x)` means `exp(x)/sum_c(x)`, which corresponds to the Eq. (5) in the paper.
    
7.  **Q: There are differences on the discrepancy loss in uncertainty calculation between paper and code. (Issue [#16](../../../issues/16))**

    **A:** Our experiments have shown that, there are not much differences in performance between using two types of loss, L1 loss and L2 loss.
    
8.  **Q: Why is there not such significant difference as (a) between MI-AOD and other methods in the first cycle in Fig. 5(b) and (c) of the paper? (Issue [#19](../../../issues/19))**

    **A:** The number of initial labeled samples is 827 in (a) but 1000 in (b).
    The number of training epochs is 26 in (a) but 300 in (b), although RetinaNet is ahead of SSD to a certain extent.
    The more data and epochs, the more fitting models, and the smaller difference between MI-AOD and other methods.
    
    Similarly, there are 2345 initial labeled samples in (c).
    And notice that MS COCO is a more challenging dataset,
    so the performances of all methods in early learning cycles are not so satisfactory with 2.0% of the labeled data,
    resulting in the little difference between the lower performances.

9.  **Q: When training the MIL classifier in Eq. (6), for an image with multiple classes, how to obtain the label of the entire image? (Issue [#20](../../../issues/20))**

    **A:** For a image with multiple classes, the label of the image will be a 1\*20 one-hot tensor (20 is the number of classes in PASCAL VOC).
    When training the classifier in the entire network, the label for each class (i.e., image label [i]) will be also trained separately.
    
10. **Q: Could you share exact numbers (mean and standard deviation) used in Fig. 5(b) for MI-AOD? (Issue [#25](../../../issues/25))**

    **A:** The numbers are as follows.
    
    |Number of Labeled Images|1k|2k|3k|4k|5k|6k|7k|8k|9k|10k|
    |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
    |Performance of MI-AOD (%) |53.62|62.86|66.83|69.33|70.80|72.21|72.84|73.74|74.18|74.91|
    |Std of MI-AOD (%)| 2.15 | 1.52 | 0.77 | 0.54 | 0.34 | 0.28 | 0.23 | 0.21 | 0.18 | 0.17 |
    
    
## Fixed Bugs and New Features
    
1.  **Q: There is not any reaction when running `./script.sh 0`. (Issues [#6](../../../issues/6) and [#13](../../../issues/13))**

    **A:** Please refer to [here](../../../#train-and-test) if you want to directly flush the running log in the terminal.
    
2.  **Q: `AttributeError: 'NoneType' object has no attribute 'param_lambda'`. (Issue [#7](../../../issues/7))**

    **A:** The bug has been fixed, please update to the latest version.
    
3.  **Q: If only a single machine and a single GPU are used for training, is distributed training still needed (like `script.sh` and `tools/dist_train.py`)? (Issue [#15](../../../issues/15))**

    **A:** Please refer to [here](../../../#train-and-test) if only using a single machine and a single GPU to train.

4.  **Q: `AssertionError: Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"`. (Issue [#17](../../../issues/17))**

    **A:** The bug has been fixed, please update to the latest version.

5.  **Q: How to run it on COCO dataset and how to modify `active_datasets.py`? (Issues [#18](../../../issues/18) and Issues [#27](../../../issues/27))**

    **A:** The code [in this repository](https://github.com/KapilM26/coco2VOC) is used for transfering the COCO json-style annotation to PASCAL VOC xml-style annotation, and COCO JPEG-style images can be directly used as PASCAL VOC JPEG-style images.

    In this way, the code for training generally remains, while the code for test can be replaced with the part of config files in mmdetection.

    Specifically, the instruction of data preparation on MS COCO is ready [here](../../../tree/master/for_coco).

6.  **Q: How to inference on single image (calculate uncertainty, or return bbox)? (Issues [#21](../../../issues/21) and [#22](../../../issues/22))**

    **A:** The new feature has been updated. Please refer to [here](../../../#train-and-test).
