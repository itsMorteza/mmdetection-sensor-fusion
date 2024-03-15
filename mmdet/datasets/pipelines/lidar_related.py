import numpy as np
import cv2
import time

from ..builder import PIPELINES
@PIPELINES.register_module()
class LoadLidarImg:
    def __int__(self):
        pass


    def __call__(self, results):
        file = results['img_info']["filename"][:6]
        #width, height = 1238, 374
        img = results['img']
        #width, height=img.size
        width, height = results['img_info']['width'], results['img_info']['height']
        lidar_map = np.fromfile(
            '/comm_dat/morteza/mmkiti/data/kitti/training/Lidar_map/' + file + '.bin', dtype=np.float64,
            count=-1).reshape([height, width, 3])
        img = results['img']

        # Here I just concatenate them, plz annote this step and use 2 resnet to extract maps and then test the op robert want
        aug_img = np.concatenate((img, lidar_map), axis=2)
        results['img'] = aug_img
        results['img_shape'] = (height, width, 6)
        results['ori_shape'] = (height, width, 6)

        return results
