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
    
    
## Fixed Bugs and New Features
    
1.  **Q: There is not any reaction when running `./script.sh 0`. (Issues [#6](../../../issues/6) and [#13](../../../issues/13))**

    **A:** Please refer to [here](../../../#train-and-test) if you want to directly flush the running log in the terminal.
    
2.  **Q: `AttributeError: 'NoneType' object has no attribute 'param_lambda'`. (Issue [#7](../../../issues/7))**

    **A:** The bug has been fixed, please update to the latest version.
    
3.  **Q: If only a single machine and a single GPU are used for training, is distributed training still needed (like `script.sh` and `tools/dist_train.py`)? (Issue [#15](../../../issues/15))**

    **A:** Please refer to [here](../../../#train-and-test) if only using a single machine and a single GPU to train.

