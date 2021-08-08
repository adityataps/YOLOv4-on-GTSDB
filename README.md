# YOLOv4 on GTSDB

This repository is a study into the Darknet YOLO framework on the German Traffic Sign Detection Benchmark (GTSDB) dataset. 

## Instructions on running this YOLOv4 model:
1. Clone this repository
2. Build the Darknet framework and its dependencies from the Darknet repository (https://github.com/AlexeyAB/darknet)
3. In the `darknet/build/darknet/x64/` folder:
    * Copy `yolov4-gtsdb_2.cfg` from this repository to `cfg/`
    * Copy `gtsdb_2.data` and `gtsdb_2.names` from this repository to `data/gtsdb_2/`
    * Copy the `annotated_imgs` folder from the P1 folder in this repository to `data/gtsdb_2/`
    * Copy the files in the `results/run_2/backup` folder from the P2 folder from this repository into `backup/`
    * Open Command Prompt and run `darknet.exe detector map data/gtsdb_2/gtsdb_2.data cfg/yolov4-gtsdb_2.cfg backup/yolov4-gtsdb_2_<DESIRED WEIGHTS CONFIGURATION HERE>.weights`


Possible issues that may arise may come from mismatched filepaths. Please double check the `annotated_imgs` files if you have any trouble with training the detector.
