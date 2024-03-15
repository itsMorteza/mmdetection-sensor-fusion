_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/kitti_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# load_from = '/home/nature/PycharmProjects/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = '/comm_dat/morteza/mmkiti/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
runner = dict(type='EpochBasedRunner', max_epochs=30)
