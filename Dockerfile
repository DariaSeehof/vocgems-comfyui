FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI

# Install ComfyUI dependencies
WORKDIR /workspace/ComfyUI
RUN pip install -r requirements.txt
RUN pip install runpod

# Download models
RUN mkdir -p /workspace/ComfyUI/models/checkpoints
RUN mkdir -p /workspace/ComfyUI/models/loras

# Download epicRealism model
RUN wget -O /workspace/ComfyUI/models/checkpoints/epicRealism.safetensors \
    "https://civitai.com/api/download/models/143906"

# Copy handler and LoRA (LoRA will be uploaded separately due to size)
COPY handler.py /workspace/handler.py

WORKDIR /workspace

CMD ["python", "-u", "handler.py"]
