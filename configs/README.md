# Variables and Parameters in the Configuration File (in MIAOD.py)

Language: [简体中文](README_cn.md) | English

<!-- TOC -->

- [Base](#base)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Results](#results)
- [Unique MI-AOD Settings](#unique-mi-aod-settings)

<!-- TOC -->

## Base

- **\_base\_** (Lines 4-7): The basic configuration files, including RetinaNet, PASCAL VOC and runtime settings.

## Datasets

- **data_root** (Line 2): Your actual directory. There is another same **data_root** in `_base_/voc0712.py` to modify.

- **data** (Lines 9-16): The data used for training, validation and test.
  
  - **test** (Line 10): The data for test. 

    - **ann_file** (Lines 11-14): The path of annotation file.
    
    - **img_prefix** (Line 15): The path of images.

## Model Training

- **model** (Line 17): The used model, including backbone, neck and head.
  
  - **bbox_head**: The bounding-box head of the model.
    
    - **C**: The number of classes in the dataset.

- **optimizer** (Line 19): The used optimizer, including learning rate, momentum and weight decay.

  - **lr**: Learning rate.

  - **momentum**: Momentum parameter.

  - **weight_decay**: Weight decay parameter.

- **optimizer_config** (Line 20): The gradient harmonizing parameter.

- **lr_config** (Line 22): The settings in learning rate.

  - **step**: The moment to drop the learning rate.

## Results

- **checkpoint_config** (Line 24): The frequency of saving models.

- **log_config** (Line 26): The frequency of printing training logs.

- **evaluation** (Line 31): The frequency of evaluating the model.

- **work_directory** (Line 48): The work directory for saving logs and files. Please refer to [here](../README.md#results) for more information.

## Unique MI-AOD Settings

- **epoch_ratio** (Line 29): The number of epochs for Label Set Training step and those for Re-weighting and Minimizing/Maximizing Instance Uncertainty steps.

- **epoch** (Line 33): The number of outer loops (i.e., all 3 training steps except the first Label Set Training step).

- **X_L_repeat** (Line 36): The repeat time for the labeled sets.

- **X_U_repeat** (Line 37): The repeat time for the unlabeled sets.

- **train_cfg** (Line 39): Some parameters for model training.

  - **param_lambda**: The regularization hyper-parameter ![lambda](http://latex.codecogs.com/gif.latex?\bg_white\lambda).

- **k** (Line 40): The hyper-parameter _k_, the number of observed top instance uncertainty in an unlabeled image for sample selection.

- **X_S_size** (Line 43): The size of ![X_S](http://latex.codecogs.com/gif.latex?\bg_white\mathit{X}_S), the newly selected sets.

- **X_L_0_size** (Line 44): The size of ![X_L^0](http://latex.codecogs.com/gif.latex?\bg_white\mathit{X}_L^0), the initial labeled set.

- **cycles** (Line 46): The number of active learning cycles.
