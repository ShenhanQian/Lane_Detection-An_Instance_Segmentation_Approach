# Lane Detection: An Instance Segmentation Approach

## Introduction

This is a PyTorch implementation and variation of the paper [《Towards End-to-End Lane Detection: An Instance Segmentation Approach》](https://arxiv.org/abs/1802.05591).

![alt text](images/lane-detection.gif "lane-detection")

## Basic Results

### Results on TuSimple Benchmark test set
|Architecture|Accuracy|FP|FN|FPS|
|-|-|-|-|-|
|FCN-Res18 |0.940|0.142|0.085|15.6|
|FCN-Res34 |0.941|0.133|0.083|14.6|
|ENet |0.937|0.149|0.093|10.8|
|ICNet |0.935|0.139|0.103|11.1|

**Note:**
- The model was trained using only TuSimple dataset, without data augmentation, the size of input images is 512*288.
- Training and testing were performed on an computer cluster with Intel Xeon CPUs(E5-2690 v4 @ 2.60GHz) and NVIDIA Titan V GPUs.


## Quick Start
### Preparation
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Download [TuSimple Benchmark](https://github.com/TuSimple/tusimple-benchmark/issues/3) dataset, and unzip the packs. The dataset structure should be as follows:
   ```
   tusimple_benchmark
    `-- |-- test_set
        |   |-- clips
        |   `-- ...
        `-- train_set
            |-- clips
            |-- label_data_xxxx.json
            |-- label_data_xxxx.json
            |-- label_data_xxxx.json
            `-- ...
   ```

3. Download checkpoint pth files from our LaneNet model zoo.
    - [onedrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianshh_shanghaitech_edu_cn/ElU__fWwiZ1DtwlhKN0hfKUBqfmqkWep3Ey93pZ7y74TIQ?e=9kR88z)
    - [百度网盘](https://pan.baidu.com/s/1LrvIVC_fdTAlYcRoEAKNLQ) (提取码: u6wa)

### Training and Testing
#### Testing on TuSimple Benchmark
```shell
python test_lanenet-tusimple_benchmark.py \
        --data_dir /path/to/test_set \
        --arch <MODEL> \
        --ckpt_path /path/to/checkpoint/file
```
- Setup testing set path using `--data_dir /path/to/test_set`
- Select network architecture by `--arch <MODEL>`, options include `fcn`, `enet`, `icnet`, and `fcn` is the default option.
- Add `--dual_decoder` to use seperate decoders for the binary segmentation branch and embedding branch. By default, these two branches shares a decoder.
- Checkpoint file should be a `*.pth` file.
- Add `--show` to display output images while testing. In each iteration, after show images ,the program pauses until a key is pressed.
- Add `--save_img` to save images into `./output/` while testing.
- Add `--ipm` to conduct **Inverse Projective Mapping(IPM)** before fitting lane curves.
- By passing `--tag <string>`, one can record experimental settings notices in a string, which will be included in the name of output directories and log files.

#### Training on TuSimple Benchmark
```shell
python train_lanenet.py \
        --data_dir /path/to/train_set \
        --arch <MODEL> \
        --ckpt_path /path/to/checkpoint/file
```
- Setup training set path using `--data_dir /path/to/train_set`, both training and validation data are loaded from this directory.
- Select network architecture by `--arch <MODEL>`, options include `fcn`, `enet`, `icnet`, and `fcn` is the default option.
- Add `--dual_decoder` to use seperate decoders for the binary segmentation branch and embedding branch. By default, these two branches shares a decoder.
- Checkpoint file should be a `*.pth` file.
- By passing `--tag <string>`, one can record experimental settings notices in a string, which will be included in the name of output directories and log files.


## Dataset division and analysis

## Architecture

## Tensorboard Summary Details

## TODO
- integrate TuSimple Bencnmark eval script with the test script
- Discuss about IPM, dataset division.
