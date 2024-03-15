import numpy as np
from rich.progress import track
import  time

import glob
import cv2


def GetMatrix(file_name):
    projection = {}
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                projection[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    v2c = projection['Tr_velo_to_cam'].reshape(3, 4)
    p2 = projection['P2'].reshape(3, 4)
    r_0 = projection['R0_rect'].reshape(3, 3)
    return v2c, p2, r_0

if __name__ == "__main__":
    lidar_file_loc = glob.glob('./kitti/training/velodyne/*.bin')
    for lidar_file in track(lidar_file_loc):
        name = lidar_file.split('.')[1].split('/')[-1]
        calib_file_loc = './kitti/training/calib/' + name + '.txt'
        image_loc = './kitti/training/image_2/' + name + '.png'
        height, width, _ = cv2.imread(image_loc).shape
        points = np.fromfile(lidar_file, dtype=np.float32, count=-1).reshape([-1, 4])
        points = points[points[:,0]>0,:]
        v2c, p2, r_0 = GetMatrix(calib_file_loc)

        points_need = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)
        v2c_arg = np.concatenate((v2c, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)
        temp = np.dot(v2c_arg, points_need.T)
        r_0_temp = np.concatenate((r_0, np.array([0, 0, 0]).reshape(1, 3)), axis=0)
        r_0_temp = np.concatenate((r_0_temp, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)
        temp = np.dot(r_0_temp, temp)
        points_in_cam2 = np.dot(p2, temp)
        points_2d = points_in_cam2[:2, :] / points_in_cam2[2, :]
        points_2d = points_2d.astype(np.int32)

        depth_map = np.array(np.zeros((height, width, 1)))
        height_map = np.array(np.zeros((height, width, 1)))
        intensity_map = np.array(np.zeros((height, width, 1)))

        for i in range(points_2d.shape[1]):
            depth = points[i][0] / 70.
            intensity = points[i][3]
            _height = 1. - (points[i][2] + 1.) / -4.

            cv2.circle(depth_map, (points_2d[0][i], points_2d[1][i]), 4, float(depth), -1)
            cv2.circle(height_map, (points_2d[0][i], points_2d[1][i]), 4, float(_height), -1)
            cv2.circle(intensity_map, (points_2d[0][i], points_2d[1][i]), 4, float(intensity), -1)

        img_map = np.concatenate((depth_map, height_map, intensity_map), axis=2)
        img_map.tofile('./kitti/training/Lidar_map/' + name + '.bin')

