# petal_ct_crop_seg

## Overview
This repository is dedicated to a method for segmenting petals from CT images of *C. japonica*.

## Usage
### Installation
This project can be set up in two ways: by installing the necessary packages directly on your system or by using Docker. Below are the instructions for both methods.

#### Direct Installation
Please refer to [get_started.md](https://mmdetection.readthedocs.io/en/v2.28.2/get_started.html#installation) for installation and dataset preparation.

#### Building with Docker

Alternatively, you can build the environment using Docker. This method ensures that the project runs in a consistent and isolated environment, regardless of your local setup. You will need Docker installed on your system. If you don't have Docker, follow the [official Docker installation guide](https://docs.docker.com/get-docker/).

Once Docker is installed, you can build the Docker image for this project using the provided Dockerfile:

```bash
# Build the Docker image from the Dockerfile
docker build -t mmdetection docker/
```

After building the image, you can run the project inside a Docker container:

```bash
# Run the project in a Docker container
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/petal_ct_crop_seg/data mmdetection
```

The `-v {DATA_DIR}:/mmdetection/petal_ct_crop_seg/data` can be optionally configured.

This command sets up the environment as specified in the Dockerfile, including the correct version of mmdetection (v2.28.2) and any other dependencies, ensuring a consistent setup across different machines.

### Train
Before you begin, make sure you navigate to the `mmdetection` directory. This is where you'll perform all subsequent commands related to this project.

```bash
cd path/to/mmdetection
```

Replace `path/to/mmdetection` with the actual path to the `mmdetection` directory on your system. This will change your current working directory to `mmdetection`, ensuring that all commands you run next are executed in the correct directory context.

```bash
# Train with a single GPU
python tools/train.py <CONFIG_FILE> [--gpu-id <GPU_ID>] [optional arguments]

# Train with multiple GPUs
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> 
```

For example, to train a Hybrid Task Cascade model with a ResNeXt101-64x4d backbone and 4 gpus, run:
```bash
tools/dist_train.sh /mmdetection/petal_ct_crop_seg/configs/htc_x101_64x4d_fpn_16x1_crop-rotate_50e_adam_coco_segm.py 4
```

### Inference
```bash
# single-gpu testing (w/o bbox result)
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval segm

# multi-gpu testing (w/o bbox result)
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval segm
```

### Petal segmentation
#### `scripts/crop_seg_mmdet.py`
This code segments petals in cropped images using a trained model.

```bash
python crop_seg_mmdet.py [-h] [--device DEVICE] [--input INPUT] [--output OUTPUT] config checkpoint start end

# positional arguments:
#   config           train config file path
#   checkpoint       checkpoint file
#   start            starting image number
#   end              ending image number

# options:
#   -h, --help       show this help message and exit
#   --device DEVICE  GPU device number. Dafault is 0.
#   --input INPUT    Directory path where images are stored. Default is /mmdetection/petal_ct_crop_seg/data/volume_1
#   --output OUTPUT  Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/crop_ct_seg
```

#### `scripts/2d_inte.py`
This code integrates the segmentation results of the cropped image.

```bash
python 2d_inte.py [-h] [--input INPUT] [--output OUTPUT] [--proc-num PROC_NUM] start end

# positional arguments:
#   start                starting image number
#   end                  ending image number

# options:
#   -h, --help           show this help message and exit
#   --input INPUT        Directory containing the segmentation results of the cropped image. Default is /mmdetection/petal_ct_crop_seg/data/crop_ct_seg
#   --output OUTPUT      Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/data/2d_inte/
#   --proc-num PROC_NUM  Number of cores used. Default is the maximum number of cores that can be used.
```

#### `scripts/3d_inte.py`
This code integrates the segmentation results of the cropped image.

```bash
python 3d_inte.py [-h] [--input INPUT] [--output OUTPUT]

# options:
#   -h, --help       show this help message and exit
#   --input INPUT    Directory where the segmentation results merged in 2D are stored. Default is /mmdetection/petal_ct_crop_seg/data/2d_inte/
#   --output OUTPUT  Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/data/3d_inte/
```

### Environment Setup

This project has been developed and tested in the following software and hardware environment.

#### Software Requirements
- **PyTorch Version**: 1.13.1
- **CUDA Version**: 11.6
- **cuDNN Version**: 8
- **Python Version**: 3.10.8
- **mmdetection Version**: 2.28.2

#### Hardware Configuration

- **GPU Server**: 
  - **GPU**: NVIDIA TITAN RTX with 24 GB of memory

- **CPU Server**: 
  - **Processor**: Intel Xeon Gold 5118
  - **Memory**: 128 GB of RAM

Please ensure that your environment matches or is compatible with these specifications to replicate our results and effectively utilize the codebase.

### Documents and Tutorials
We list some documents and tutorials from [MMDetection](https://github.com/open-mmlab/mmdetection), which may be helpful to you.

- [MMDetection 2.28.2 documentation](https://mmdetection.readthedocs.io/en/v2.28.2/)
- [TUTORIAL 1: LEARN ABOUT CONFIGS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/config.html)
- [TUTORIAL 2: CUSTOMIZE DATASETS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/customize_dataset.html)
- [TUTORIAL 7: FINETUNING MODELS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/finetune.html)

## Citation

## License
