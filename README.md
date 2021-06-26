# Pytorch Complex Yolo - Yolov4


Complete but Unofficial PyTorch Implementation of [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/pdf/1803.06199.pdf) with YoloV3
Code was tested with following specs:
- Code was tested on Windows 10

## 1. Installation
First, clone or download this GitHub repository.
Install requirements and download from official darknet weights:
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

# yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# yolov4-tiny
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

Or you can download darknet weights from my google drive:

https://drive.google.com/drive/folders/1w4KNO2jIlkyzQgUkkcZ18zrC8A1Nqvwa?usp=sharing

## 2. Pretrained weights

You can download darknet weights from my google drive:

- [x] Complex Yolo v4:

https://drive.google.com/file/d/1jVNRuencSHRtjDQcWySVDGSI_bqfKfCP/view?usp=sharing

- [x] Complex Yolo v4 - tiny:

      On training


## 3. Quick start

### 3.1. Download pretrained weights 

### 3.2. Dataset shall contains shall contains belows.
- [x] `detect_1` folder shall contain folders and contents.
- [x] `ImageSets` folder shall contain *.txt files.

      ${ROOT}
      ├── dataset/
      │    ├── classes.names
      │    └── kitti/
      │          ├── classes_names.txt
      │          ├── detect_1/    <-- for detection test
      │          │    ├── calib/
      │          │    ├── image_2/
      │          │    └── velodyne/
      │          ├── ImageSets/ 
      │          │    ├── detect_1.txt
      │          │    ├── detect_2.txt
      │          │    ├── sample.txt
      │          ├── sampledata/ 
      │          │    ├── image_2/
      │          │    ├── calib/
      │          │    ├── label_2/
      │          │    └── velodyne/



### 3.3 Test [without downloading dataset] 

- [x] Detection test for `Yolo v3` with `detect_1` folder.

       $ python detection.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth
       $ python detection.py
      
- [x] Detection test for `Yolo v3-tiny` with `detect_1` folder.

       On training : it will be updated soon!  
             
       
### 3.4 Demo Video 
- [x] One side detection demo.

![pred_complex_yolo_v4](./asset/pred_complex_yolo_v4.gif)

## 4. Data Preparation from KITTI

You can see `sampledata` folder in `dataset/kitti/sampledata` directory which can be used for testing this project without downloading KITTI dataset. However, if you want to train the model by yourself and check the mAP in validation set just follow the steps below.

#### Download the [3D KITTI detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 
1. Camera calibration matrices of object data set (16 MB)
2. Training labels of object data set (5 MB)
3. Velodyne point clouds (29 GB)
4. Left color images of object data set (12 GB)

Now you have to manage dataset directory structure. Place your dataset into `dataset` folder. Please make sure that you have the dataset directory structure as follows. 


The `train/valid` split of training dataset as well as `sample` and `test` dataset ids are in `dataset/kitti/ImageSets` directory. From training set of 7481 images, 6000 images are used for training and remaining 1481 images are used for validation. The mAP results reported in this project are evaluated into this valid set with custom mAP evaluation script with 0.5 iou for each object class. 


## 5. Data Preparation from my google drive

If you download the dataset from `the KITTI Vision Benchmark Suite`, it is difficult to build detect dataset.
So, I recommend you to download dataset from my google drive.

https://drive.google.com/file/d/12Fbtxo2F8Nn9dHDV9UPnovVaCf-Uc89R/view?usp=sharing

### 5.1 Dataset folder structure shall be

```
${ROOT}
├── dataset/
│    ├── classes.names
│    └── kitti/
│          ├── classes_names.txt
│          ├── detect_1/    <-- for detection test
│          │    ├── calib/
│          │    ├── image_2/
│          │    └── velodyne/
│          ├── detect_2/    <-- for detection test
│          │    ├── calib/
│          │    ├── image_2/
│          │    └── velodyne/
│          ├── ImageSets/ 
│          │    ├── detect_1.txt
│          │    ├── detect_2.txt
│          │    ├── sample.txt
│          │    ├── test.txt
│          │    ├── train.txt
│          │    ├── val.txt
│          │    └── valid.txt
│          ├── sampledata/ 
│          │    ├── image_2/
│          │    ├── calib/
│          │    ├── label_2/
│          │    └── velodyne/
│          ├── training/    <-- 7481 train data
│          │    ├── image_2/  <-- for visualization
│          │    ├── calib/
│          │    ├── label_2/
│          │    └── velodyne/
│          └── testing/     <-- 7580 test data
│                ├── image_2/  <-- for visualization
│                ├── calib/
│                └── velodyne/
```


## 6. Train

- [x] `Complex Yolo v4` training from `pretrained weight`.

       $ python train.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth --save_path checkpoints/Complex_yolo_yolo_v4.pth
       $ python train.py
       
- [x] `Complex Yolo v4` training from `darknet weight`.

       $ python train.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/yolov4.weights --save_path checkpoints/Complex_yolo_yolo_v4.pth
    
- [x] `Complex Yolo v4-tiny` training from `pretrained weight`.

       $ python train.py --model_def config/cfg/complex_yolov4_tiny.cfg --pretrained_path checkpoints/yolov4-tiny.weights --save_path checkpoints/Complex_yolo_yolo_v4_tiny.pth --batch_size 8 
    
- [x] `Complex Yolo v4-tiny` training from `darknet weight`.

       $ python train.py --model_def config/cfg/complex_yolov4_tiny.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4_tiny.pth --save_path checkpoints/Complex_yolo_yolo_v4_tiny.pth --batch_size 8 
    
## 7. Evaluation

- [x] `Complex Yolo v3` evaluation.

       $ python eval_mAP.py 
       $ python eval_mAP.py --model_def config/complex_yolov3.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v3.pth --batch_size 2

```
    Detecting objects: 100%|███████████████████████████████████████████████████████████████████████| 741/741 [02:44<00:00,  4.51it/s]
    Computing AP: 100%|███████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 107.44it/s]

    Done computing mAP...

        >>>      Class 0 (Car): precision = 0.9052, recall = 0.9866, AP = 0.9790, f1: 0.9441
        >>>      Class 1 (Ped): precision = 0.6389, recall = 0.9317, AP = 0.8272, f1: 0.7580
        >>>      Class 2 (Cyc): precision = 0.7927, recall = 0.9524, AP = 0.9013, f1: 0.8652

    mAP: 0.9025
``` 
    
- [x] `Complex Yolo v3 - tiny` evaluation.

       $ python eval_mAP.py --model_def config/complex_yolov3_tiny.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v3_tiny.pth --batch_size 8
    
```
    Detecting objects: 100%|████████████████████████████████████████████████████████████████████████| 186/186 [01:36<00:00,  1.93it/s]
    Computing AP: 100%|████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 103.71it/s]

    Done computing mAP...

        >>>      Class 0 (Car): precision = 0.8858, recall = 0.9706, AP = 0.9592, f1: 0.9263
        >>>      Class 1 (Ped): precision = 0.4882, recall = 0.7662, AP = 0.4929, f1: 0.5964
        >>>      Class 2 (Cyc): precision = 0.6042, recall = 0.9451, AP = 0.7875, f1: 0.7371

    mAP: 0.7465
```


## Credit

1. Complex-YOLO: https://arxiv.org/pdf/1803.06199.pdf

YoloV3 Implementation is borrowed from:
1. https://github.com/eriklindernoren/PyTorch-YOLOv3

Point Cloud Preprocessing is based on:  
1. https://github.com/skyhehe123/VoxelNet-pytorch
2. https://github.com/dongwoohhh/MV3D-Pytorch


## Folder structure

```
${ROOT}
├── detection.py
├── detection_both_side.py
├── eval_mAP.py
├── README.md 
├── train.py
├── checkpoints/ 
│    ├── Complex_yolo_yolo_v3.pth
│    ├── Complex_yolo_yolo_v3_tiny.pth
│    ├── yolov3.weights
│    └── yolov3-tiny.weights
├── config/ 
│    ├── complex_yolov3.cfg
│    ├── complex_yolov3_tiny.cfg
│    ├── complex_yolov4.cfg
│    ├── complex_yolov4_tiny.cfg
│    └── train_config.py
├── data_process/ 
│    ├── config.py
│    ├── kitti_aug_utils.py
│    ├── kitti_bev_utils.py
│    ├── kitti_dataset.py
│    ├── kitti_utils.py
│    └── kitti_yolo_dataset.py
├── dataset/
│    ├── classes.names
│    └── kitti/
│          ├── classes_names.txt
│          ├── detect_1/    <-- for detection test
│          │    ├── calib/
│          │    ├── image_2/
│          │    └── velodyne/
│          ├── detect_2/    <-- for detection test
│          │    ├── calib/
│          │    ├── image_2/
│          │    └── velodyne/
│          ├── ImageSets/ 
│          │    ├── detect_1.txt
│          │    ├── detect_2.txt
│          │    ├── sample.txt
│          │    ├── test.txt
│          │    ├── train.txt
│          │    ├── val.txt
│          │    └── valid.txt
│          ├── sampledata/ 
│          │    ├── image_2/
│          │    ├── calib/
│          │    ├── label_2/
│          │    └── velodyne/
│          ├── training/    <-- 7481 train data
│          │    ├── image_2/  <-- for visualization
│          │    ├── calib/
│          │    ├── label_2/
│          │    └── velodyne/
│          └── testing/     <-- 7580 test data
│                ├── image_2/  <-- for visualization
│                ├── calib/
│                └── velodyne/
├── logs/ 
├── models/ 
│    ├── models.py
│    └── yolo_layer.py
└── utils/ 
      ├── mayavi_viewer.py
      ├── train_utils.py
      └── utils.py

```
