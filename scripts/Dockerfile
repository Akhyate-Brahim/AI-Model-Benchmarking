FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

VOLUME model_cache:/root/.cache/torch

COPY imagenet1000_clsidx_to_labels.txt .
COPY cnn_comparison.py .
COPY imagenet-samples/ ./imagenet-samples/

CMD ["python", "cnn_comparison.py"]