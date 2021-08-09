# Please change the dataset directory to your actual directory
data_root = '$YOUR_DATASET_PATH/VOCdevkit/'

_base_ = [
    './_base_/ssd300.py', './_base_/voc0712.py',
    './_base_/default_runtime.py'
]
# We use PASCAL VOC 2007+2012 trainval sets to train, so we also use them to select the informative samples.
# from v.2.3.0 ssd config file
model = dict(bbox_head=dict(C=20, anchor_generator=dict(basesize_ratio_range=(0.2, 0.9))))
dataset_type = 'VOCDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset', times=1, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(
        type='VOCDataset',
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=test_pipeline
    )
)
# The initial learning rate, momentum, weight decay can be changed here.
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# The moment when the learning rate drops can be changed here.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1])
# The frequency of saving models can be changed here.
checkpoint_config = dict(interval=1)
# The frequency of printing training logs (including progress, learning rate, time, loss, etc.) can be changed here.
log_config = dict(interval=50)
# The number of epochs for Label Set Training step and those for Re-weighting and Minimizing/Maximizing Instance
# Uncertainty steps can be changed here.
epoch_ratio = [5, 1]
# The frequency of evaluating the model can be changed here.
evaluation = dict(interval=epoch_ratio[0], metric='mAP')
# The number of outer loops (i.e., all 3 training steps except the first Label Set Training step) can be changed here.
epoch = 2
# The repeat time for the labeled sets and unlabeled sets can be changed here.
# The number of repeat times can be equivalent to the number of actual training epochs.
X_L_repeat = 16
X_U_repeat = 16
# The hyper-parameters lambda and k can be changed here.
train_cfg = dict(param_lambda = 0.5)
k = 10000
# The size of the initial labeled set and the newly selected sets after each cycle can be set here.
# Note that there are 16551 images in the PASCAL VOC 2007+2012 trainval sets.
X_S_size = 1000
X_L_0_size = 1000
# The active learning cycles can be changed here.
cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# The work directory for saving logs and files can be changed here. Please refer to README.md for more information.
work_directory = './work_dirs/MI-AOD_SSD'

