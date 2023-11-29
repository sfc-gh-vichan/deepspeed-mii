FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /

COPY relax/images/deepspeed_mii/requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y git

# Commits pinned due to unstable releases
RUN pip install git+https://github.com/microsoft/deepspeed.git@2afa1c7f2f961ef18042a88467ff5d3373c22c07
RUN pip install git+https://github.com/Microsoft/DeepSpeed-MII.git@a5d2f26351f478238491bd14715aea34b73a1e6c

COPY relax/deepspeed_mii /relax/deepspeed_mii/
