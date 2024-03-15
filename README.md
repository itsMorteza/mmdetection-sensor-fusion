# Sensor Fusion Operators for the OpenMMLab Detection Toolbox
## Introduction
This repository contains the sensor fusion operators for the OpenMMLab Detection Toolbox. The sensor fusion operators are used to fuse the detection results from different sensors, such as RGB camera, LiDAR  (early and mid level fusion). 
## Installation
for installation follow the instructions in the [get_started.md](./docs/en/get_started.md) file.
## Supported Dataset and Data Preprocessing
  Please make sure the folder organized like follows before run Preprocess_lidar_map.py:
    ```
    ├── cfg
    ├── data
    │   ├── Preprocess_lidar_map.py
    │   ├── kitti
    │   │   ├── training
    │   │   ├── testing
    │   │   ├── ImageSets
    ```
    Then run the generation script
    ```
    python Preprocess_lidar_map.py
    ```
## train
  ```
  python tools/train.py ${CONFIG_FILE} 
  ```
## test
  ```
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} 
  ```
## Citation
If you use this toolbox or benchmark in your research, please cite this project.
```
@inproceedings{pasandi2022sensor,
  title={Sensor fusion operators for multimodal 2d object detection},
  author={Pasandi, Morteza Mousa and Liu, Tianran and Massoud, Yahya and Lagani{\`e}re, Robert},
  booktitle={International Symposium on Visual Computing},
  pages={184--195},
  year={2022},
  organization={Springer}
}

@article{mousa2023rgb,
  title={RGB-LiDAR fusion for accurate 2D and 3D object detection},
  author={Mousa-Pasandi, Morteza and Liu, Tianran and Massoud, Yahya and Lagani{\`e}re, Robert},
  journal={Machine Vision and Applications},
  volume={34},
  number={5},
  pages={86},
  year={2023},
  publisher={Springer}
}
```

## TODO
- [1] Add results and vizualization
- [2] Add the pretrained models
- [3] Append the other sensor fusion operators

## License
the source code is released under the [Apache 2.0 license](./LICENSE).