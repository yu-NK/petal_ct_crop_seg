ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#setup environment
ENV FORCE_CUDA="1" TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" PATH="/usr/local/cuda/bin:$PATH" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" MMCV_WITH_OPS="1"
RUN apt update && apt install -y git libgl1 libglib2.0-0 ffmpeg libturbojpeg libsm6 libxext6 ninja-build libxrender-dev && apt clean && rm -rf /var/lib/apt/lists/*

#install mmcv
RUN git clone https://github.com/open-mmlab/mmcv.git -b v1.7.1 /mmcv
WORKDIR /mmcv
RUN pip install --no-cache-dir -e .[all] -v


#install mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git -b v2.28.2 /mmdetection
WORKDIR /mmdetection
RUN pip install --no-cache-dir -e . -v

#install petal_ct_crop_seg
RUN git clone https://github.com/yu-NK/petal_ct_crop_seg.git

# if you need, you write the below code
RUN pip install --no-cache-dir albumentations
RUN apt-get update && apt-get install -y git vim wget curl zip unzip
RUN pip install jupyterlab tqdm numpy matplotlib opencv-python scikit-image scikit-learn pillow ipywidgets yapf==0.40.1