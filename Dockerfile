FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    git wget curl ffmpeg libsm6 libxext6 libglib2.0-0 \
    libgomp1 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install runpod aiohttp

WORKDIR /app
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /app/ComfyUI
RUN pip install -r requirements.txt

RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    custom_nodes/ComfyUI-LTXVideo && \
    cd custom_nodes/ComfyUI-LTXVideo && pip install -r requirements.txt

RUN mkdir -p /models/checkpoints /models/vae /models/text_encoders
RUN ln -sf /models/checkpoints /app/ComfyUI/models/checkpoints && \
    ln -sf /models/vae /app/ComfyUI/models/vae && \
    ln -sf /models/text_encoders /app/ComfyUI/models/text_encoders

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
