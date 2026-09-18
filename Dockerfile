FROM python:3.14-slim

# "video" is what lets NVENC see libnvidia-encode from the host driver.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Distro ffmpeg builds are inconsistent about NVENC; this one always has it.
RUN curl -fsSL -o /tmp/ffmpeg.tar.xz \
        https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && install -m755 /tmp/ffmpeg-*/bin/ffmpeg /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rife.py esrgan.py enhance.py download_models.py ./

# Baked into the image so a cold RunPod start does not re-download ~95 MB.
# Use --all here if you also want the alternative RIFE versions in the image.
RUN python download_models.py

ENTRYPOINT ["python", "enhance.py"]
CMD ["--help"]
