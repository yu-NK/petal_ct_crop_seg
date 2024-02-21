# petal_ct_crop_seg

## Overview

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


### Environment
This project was developed and tested with Python version 3.10.8.
The implementation uses mmdetection version 2.28.2. 

### Documents and Tutorials
We list some documents and tutorials from [MMDetection](https://github.com/open-mmlab/mmdetection), which may be helpful to you.

- [MMDetection 2.28.2 documentation](https://mmdetection.readthedocs.io/en/v2.28.2/)
- [TUTORIAL 1: LEARN ABOUT CONFIGS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/config.html)
- [TUTORIAL 2: CUSTOMIZE DATASETS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/customize_dataset.html)
- [TUTORIAL 7: FINETUNING MODELS](https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/finetune.html)

## Citation

## License
