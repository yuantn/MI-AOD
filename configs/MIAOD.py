# Please change the dataset directory to your actual directory
data_root = '$YOUR_DATASET_PATH/VOCdevkit/'

_base_ = [
    './_base_/retinanet_r50_fpn.py', './_base_/voc0712.py',
    './_base_/default_runtime.py'
]
# We use PASCAL VOC 2007+2012 trainval sets to train, so we also use them to select the informative samples.
data = dict(
    test=dict(
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'])
)
model = dict(bbox_head=dict(C=20))
# The initial learning rate, momentum, weight decay can be changed here.
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# The moment when the learning rate drops can be changed here.
lr_config = dict(policy='step', step=[2])
# The frequency of saving models can be changed here.
checkpoint_config = dict(interval=1)
# The frequency of printing training logs (including progress, learning rate, time, loss, etc.) can be changed here.
log_config = dict(interval=50)
# The number of epochs for Label Set Training step and those for Re-weighting and Minimizing/Maximizing Instance
# Uncertainty steps can be changed here.
epoch_ratio = [3, 1]
# The frequency of evaluating the model can be changed here.
evaluation = dict(interval=epoch_ratio[0], metric='mAP')
# The number of outer loops (i.e., all 3 training steps except the first Label Set Training step) can be changed here.
epoch = 2
# The repeat time for the labeled sets and unlabeled sets can be changed here.
# The number of repeat times can be equivalent to the number of actual training epochs.
X_L_repeat = 2
X_U_repeat = 2
# The hyper-parameters lambda and k can be changed here.
train_cfg = dict(param_lambda = 0.5)
k = 10000
# The size of the initial labeled set and the newly selected sets after each cycle can be set here.
# Note that there are 16551 images in the PASCAL VOC 2007+2012 trainval sets.
X_S_size = 16551//40
X_L_0_size = 16551//20
# The active learning cycles can be changed here.
cycles = [0, 1, 2, 3, 4, 5, 6]
# The work directory for saving logs and files can be changed here. Please refer to README.md for more information.
work_directory = './work_dirs/MI-AOD'

