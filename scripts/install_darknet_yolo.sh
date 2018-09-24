rm -rf /tmp/YOLO3
mkdir /tmp/YOLO3
git clone https://github.com/madhawav/YOLO3-4-Py /tmp/YOLO3
export GPU=1
export OPENCV=1
export CUDA_HOME=/usr/local/cuda-9.0
pipenv install /tmp/YOLO3