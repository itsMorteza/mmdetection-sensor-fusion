_base_ = [ '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/kitti_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# load_from = '/home/nature/PycharmProjects/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=14,
        num_stages=3,
        in_channels=3,
        base_channels=32,
        strides=(2, 2, 2),
        dilations=(1, 1,  1),
        out_indices=(0,1,2 ),
        stage_with_dcn=(False, False, False),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True),        
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128],
        out_channels=256,
        num_outs=5),
    roi_head=dict(bbox_head=dict(num_classes=3)))
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
