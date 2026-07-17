# CPU-only by construction. Docker Desktop on Apple Silicon exposes no Metal/MPS device,
# so this image cannot use the GPU no matter what OT_DEVICE says - hence the hard pin below.
FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .
# linux-aarch64 has no MPS; conda resolves the CPU builds of torch/torchvision here.
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "objecttracker", "/bin/bash", "-c"]

COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Bake the weights in so the container does not download on first request.
RUN conda run -n objecttracker python -c "\
from ultralytics import YOLO; YOLO('yolov8n.pt'); \
import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

ENV OT_IN_DOCKER=1 \
    OT_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "objecttracker", \
     "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
