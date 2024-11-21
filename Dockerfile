# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV MAX_JOBS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    pkg-config \
    cmake \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and related packages
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install fvcore and iopath
RUN pip3 install 'fvcore>=0.1.5' 'iopath>=0.1.7' typeguard

# Install CUB
RUN wget https://github.com/NVIDIA/cub/archive/refs/tags/2.1.0.tar.gz \
    && tar xzf 2.1.0.tar.gz \
    && mv cub-2.1.0 cub \
    && rm 2.1.0.tar.gz
ENV CUB_HOME="/cub"

# Clone and install PyTorch3D
RUN git clone https://github.com/facebookresearch/pytorch3d.git \
    && cd pytorch3d \
    # && pip3 install -e .
    && pip3 install . \
    && cd .. && rm -rf pytorch3d

WORKDIR /content

# Exactly match Colab steps
RUN git clone https://github.com/tencent/Hunyuan3D-1

WORKDIR /content/Hunyuan3D-1

# Install Python packages exactly as in Colab
RUN pip3 install packaging
RUN pip3 install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install flash_attn huggingface_hub[cli] onnxruntime

# Create weights directory and download models exactly as in Colab
RUN mkdir weights && \
    huggingface-cli download tencent/Hunyuan3D-1 --local-dir ./weights && \
    mkdir weights/hunyuanDiT && \
    huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled --local-dir ./weights/hunyuanDiT

# Set permissions as in Colab
RUN chmod +x env_install.sh app.py

# Run environment setup
RUN bash env_install.sh

# Update server configuration exactly as in Colab
RUN sed -i.bak 's/demo.launch(server_name=CONST_SERVER, server_port=CONST_PORT)/demo.launch(server_name="0.0.0.0", server_port=7860)/' app.py

# Add device check from Colab
RUN echo 'import torch; device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Using device:", device)' > check_device.py

# Set default command to match Colab
CMD ["python3", "app.py"]
