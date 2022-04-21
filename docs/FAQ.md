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
- [Custom Modifications](#custom-modifications)

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
    
    **A:** Please refer to the step 5 [here](installation.md#environment-installation) to install build requirements and install and compile MMDetection.

4.  **Q: After the model normally trained for a cycle and the weight file `*.npy` generated, it suddenly reported an error when entering the next cycle: `RuntimeError: CUDA error: no kernel image is available for execution on the device`. (Issue [#36](../../../issues/36))**

    **A:** The training has not started, the `*0.npy` is generated before the first training cycle.
    The reason of the error is that the CUDA runtime version and compiler version in mmdet do not match.

5.  **Q: `ModuleNotFoundError: No module named 'torchvision.models.segmentation.data_loader'`. (Issue [#37](../../../issues/37))**

    **A:** Please re-install PyTorch, torchvision adapted to your current CUDA version.


## Training and Test

1.  **Q: `AttributeError: 'Tensor' object has no attribute 'isnan'`. (Issues [#2](../../../issues/2) and [#9](../../../issues/9))**

    **A:** Option 1. Re-install the **Pytorch==1.6.0** and **TorchVision==0.7.0** with the [PyTorch official instructions](https://pytorch.org/get-started/previous-versions/#v160).
    
    Option 2. Check the lines of `AttributeError`, and replace the `if value.isnan()` with `if value != value` ( considering that only nan != nan).
    
    The error must be in the Line 483 and 569 of the `./mmdet/models/dense_heads/MIAOD_head.py`.
    
2.  **Q: There is not any reaction when running `./script.sh 0`. (Issues [#6](../../../issues/6) and [#13](../../../issues/13))**

    **A:** When running `script.sh`, the code is executed in the background.
    You can view the output log by running this command in the root directory: `vim log_nohup/nohup_0.log`.
    
    There is another solution to flush the logs in the terminal [in another section](#fixed-bugs-and-new-features).
    
3.  **Q: `StopIteration`. (Issues [#7](../../../issues/7#issuecomment-823068004), [#11](../../../issues/11) and [#31](../../../issues/31))**

    **A:** __If the model is trained on single GPU:__
    
    Please increase the number of training data. We recommend to use at least 5% of the images (16551 * 5% = 827 images) using RetinaNet on PASCAL VOC.
    
    __If the model is trained on multiple GPUs:__
    
    Thanks for the solution from [@KevinChow](https://github.com/kevinchow1993).
    
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
    
4.  **Q: Validation error: `TypeError: 'DataContainer' object is not subscriptable`. (Issue [#14](../../../issues/14))**

    **A:** In `get_bboxes` function of `mmdet/models/dense_heads/MIAOD_head.py`, please change
    
    ```python
    img_shape = img_metas[img_id]['img_shape']
    ```
    
    to
    
    ```python
    img_shape = img_metas.data[0]
    ```
    
    Note: You only need to make changes when you encounter this problem, usually it won't occur on a GPU environment.

5.  **Q: What is `$CONFIG_PATH` and `$CKPT_PATH` in `python tools/test.py $CONFIG_PATH $CKPT_PATH`? (Issue [#17](../../../issues/17))**

    **A:** Please refer to [here](../../../#train-and-test) for explanation. That is:
    
    > where $CONFIG_PATH should be replaced by the path of the config file in the configs folder (usually it would be configs/MIAOD.py)
    
    > $CKPT_PATH should be replaced by the path of the checkpoint file (*.pth) in the work_dirs folder after training.
    
6.  **Q: In `tools/train.py`, is it first trained on the labeled dataset? What is the purpose? (Issue [#25](../../../issues/26))**

    **A:** It is necessary to train on the labeled set for the first and last epochs to ensure the stability of the training model.
    
7.  **Q: For the unlabeled set, why are the operations on GT information (ie, `gt_bboxes` and `gt_labels`) also involved in lines 70-74 of `epoch_based_runner.py`? (Issues [#28](../../../issues/28) and [#29](../../../issues/29))**

    **A:** These lines are to remove the localization information of the images in the unlabeled set.
    In this way, when calculating the loss on the unlabeled set, we can know the data source without backward propagating the gradient.
    In fact, the GT information has not been used.

8.  **Q: What does `epoch_ratio = [3, 1]` mean in `configs/MIAOD.py`? Can I change it to `epoch_ratio = [3, 0]`? (Issue [#31](../../../issues/31#issuecomment-881190530))**

    **A:** Please refer to [here](../../../tree/master/configs#unique-mi-aod-settings) for config explanations.
    
    If you change it to [3, 0], there will not be maximizing and minimizing uncertainty.

9.  **Q: `IndexError: index 0 is out of bounds for dimension 0 with size 0`. (Issue [#31](../../../issues/31#issuecomment-881223658), [#39](../../../issues/39) and [#40](../../../issues/39))**

    **A:** A possible solution can be changing
    
    ```python
    if y_loc_img[0][0][0] < 0:
    ```

    in Line 479 in `L_wave_min` in `mmdet/models/dense_heads/MIAOD_head.py` to:

    ```python
    if y_loc_img[0][0] < 0:
    ```
    
    If it doesn't work, please insert an exception detection or use IDEs like PyCharm to set a breakpoint in the error line, and print `y_loc_img[0][0][0]` and `y_loc_img[0][0]` only when the error occurs to find if `y_loc_img` is an empty list.
    
    If it is, please re-prepare the annotations of the datasets.
    
    If you are training on a custom dataset, please refer to the notes in [Question 3 in Custom Modifications](#custom-modifications).

10. **Q: How to save the trained model for each cycle? (Issue [#32](../../../issues/32))**

    **A:** At present, this repository can save the trained model for each cycle.
    
    [Here](../../../#results) is a link to the cloud drive of an example output folder.

11. **Q: When using `tools/test.py` for test, do I need to change the `data.test.ann_file` in `config` to the true test set (instead of using _trainval_ data to calculate uncertainty)? (Issue [#32](../../../issues/32#issuecomment-879984647))**

    **A:** No, in this repository, we use the _test_ set for test, but we use `data.val` in `config`. Please refer to [here](../configs/_base_/voc0712.py).

12. **Q: What does `y_loc_img[0][0][0] < 0` mean? (Issue [#40](../../../issues/40))**

    **A:** It means that the current data batch is unlabeled, because we have set all the coordinates of the bounding box of the unlabeled data to -1 in Lines 70-74 in `epoch_based_runner.py`.
    
    In addition, thanks to [@horadrim-coder](https://github.com/horadrim-coder) for an alternative solution, which can avoid the error `IndexError: index 0 is out of bounds for dimension 0 with size 0`:
    
    1. Add the following method to `epoch_based_runner.py`:
    
    ```python
    def _add_dataset_flag(self, X, is_unlabeled):
        or _img_meta in X['img_metas'].data[0]:
         _img_meta.update({'is_unlabeled': is_unlabeled})
    ```
    
    2. Add these codes in the following lines in `epoch_based_runner.py`:

    ```python
    Line 31: `self._add_dataset_flag(X_L, is_unlabeled=False)`
    Line 60: `self._add_dataset_flag(X_L, is_unlabeled=False)`
    Line 79: `self._add_dataset_flag(X_U, is_unlabeled=True)`
    ```
    
    3. Replace `y_loc_img[0][0][0] < 0` with `img_metas[0]['is_unlabeled']` in `MIAOD_head.py` (_e.g._, Lines 479 and 565).

13. **Q: Unable to download pre-trained SSD model. (Issue [#42](../../../issues/42))**

    **A:** The pre-trained SSD model [link](https://download.openmmlab.com/pretrain/third_party/vgg16_caffe-292e1171.pth) is available in [the latest version of mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json#L2).
    After downloading, you can move it from the download folder to the default cache folder of the pre-trained model:
    
    ```bash
    mv vgg16_caffe-292e1171.pth ~/.cache/torch/hub/checkpoints
    ```


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

11. **Q: What is the pre-trained model corresponding to the SSD network? (Issue [#33](../../../issues/33))**

    **A:** The model is the `vgg16_caffe model` provided by `open-mmlab`.

12. **Q: The epoch number of the SSD code is inconsistent with the 100 described in the paper. (Issue [#33](../../../issues/33#issuecomment-895883958))**

    **A:** I have not declared in the paper that the epoch number of SSD is 100, but 300 (240+60, as described in section 4.1 of the paper).

13. **Q: What are the experimental settings for the results of Random? (Issue [#33](../../../issues/33#issuecomment-895883958))**

    **A:** I removed all the training process on the unlabeled set, and selected the image randomly.

14. **Q: The number of labeled images increases by 2k (not 1k in the paper) each cycle according to the output log `...Epoch [1][50/2000]...`. (Issue [#38](../../../issues/38))**

    **A:** The `2000` in the log is `X_L_0_size * X_L_repeat / samples_per_gpu = 1000 * 16 / 8`, and the number of added labeled images should be the shape of `X_L_0.npy` in the output directory, which is `(1000,)`.

15. **Q: How are the heatmaps drawn in the paper? (Issue [#41](../../../issues/41))**

    **A:** We respectively calculated the `l_dis`, `{y^}^cls` and `{l~}_dis` of the two classifiers on each anchor, and fill the anchors corresponding to these values with the color of the heatmap.
    
    The larger the value, the more red. The smaller the value, the more blue-violet.
    
    Finally, we add the heat map of multiple anchors to the original image in a certain proportion to get Fig. 6 in the paper.


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

7.  **Q: Is there a code to use SSD network? (Issue [#33](../../../issues/33))**

    **A:** The code has been updated. Please refer to [here](../../../#data-preparation) for the instruction of using the SSD detector.


## Custom Modifications

1.  **Q: I want to run MI-AOD with other data, which files should I modify? (Issue [#13](../../../issues/13#issuecomment-845709365))**

    **A:** You should only modify `configs/MIAOD.py` if you can convert your other training and test data into PASCAL VOC format. It contains all parameters and settings.
    
2.  **Q: When training on custom dataset (only 1 foreground class), why is l_imgcls always 0 during training? (Issues [#23](../../../issues/23)， [#24](../../../issues/24)， [#34](../../../issues/34) and [#35](../../../issues/35))**

    **A:** To avoid that, you can create another class without any corresponding image in the dataset.

3.  **Q: If the completely unlabeled data is used as the unlabeled set, what needs to be modified? (Issue [#29](../../../issues/29#issuecomment-871210792))**

    **A:** If the completely unlabeled data is used as the unlabeled set, you can add any bounding box to the annotation of the unlabeled data arbitrarily.
    The annotation format of the bounding box needs to be consistent with that of other labeled data.
    After that, just add the file name to the txt index of the unlabeled data set.

4.  **Q: `TypeError: init() missing 1 required positional argument: 'input_size'` (when changing backbone RetinaNet to custom SSD). (Issue [#30](../../../issues/30))**

    **A:** Please add `input_size=input_size` in the dict `model.backbone` in your custom configuration file `ssd300.py`.
    To avoid more potential problems, please customize any files on the MMDetection in [version 2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0/) but not [the latest version](https://github.com/open-mmlab/mmdetection/).

5.  **Q: When testing the code with my own dataset (pedestrian class only), the loss of `l_det_loc` and `L_det` is nan. (Issue [#34](../../../issues/34) and [#35](../../../issues/35))**

    **A:** Please check whether there is any problem with the bounding box annotation of your dataset.
