data_root_coco = '$YOUR_DATASET_PATH/coco/'
data_root_voc = '$YOUR_DATASET_PATH/coco2voc/'

# same base
_base_ = [
    './_base_/retinanet_r50_fpn.py',
    './_base_/default_runtime.py'
]

# COCO dir: val-img, val-xml
# VOC dir: traintest-img(-s), traintest-xml
# coco_detection.py + voc0712.py
dataset_type_voc = 'VOCDataset'
dataset_type_coco = 'CocoDataset'
# same
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# COCO
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# COCO
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # VOC
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type_voc,
            ann_file=data_root_voc + 'ImageSets/Main/trainval.txt',
            img_prefix=data_root_voc,
            pipeline=train_pipeline)),
    # COCO
    val=dict(
        type=dataset_type_coco,
        ann_file=data_root_coco + 'annotations/instances_val2017.json',
        img_prefix=data_root_coco + 'val2017/',
        pipeline=test_pipeline),
    # VOC
    test=dict(
        type=dataset_type_voc,
        ann_file=data_root_voc + 'ImageSets/Main/trainval.txt',
        img_prefix=data_root_voc,
        pipeline=test_pipeline))

# schedule_1x.py + retinanet_r50_fpn_1x_coco.py + retinanet_r50_fpn_1x_voc0712.py
# my optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
# same optimizer_config
optimizer_config = dict(grad_clip=None)
# my learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2])
# voc model but coco num_classes
model = dict(bbox_head=dict(C=80))

# # fpn config
# test_cfg = dict(
#     nms_pre=1000,
#     min_bbox_size=0,
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.5),
#     max_per_img=100)

# my min-max config
epoch_ratio = [3, 1]
# my evaluation based on COCO metric (coco_detection.py + voc0712.py)
evaluation = dict(interval=epoch_ratio[0], metric='bbox')

# same default_runtime.py config
checkpoint_config = dict(interval=1)
log_config = dict(interval=50)
train_cfg = dict(param_lambda = 0.5)
k = 10000

epoch = 2
# my config
X_L_repeat = 2
X_U_repeat = 2
num_samples = 117266
X_S_size = num_samples//50
X_L_0_size = num_samples//50
subset_p = 5
# subset_p = 100  # debug
work_directory = './work_dirs/MI-AOD_COCO'
cycles=[0, 1, 2, 3, 4]
