FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libglfw3 libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    tensorflow==2.13.0 \
    "highway-env==1.8.2" \
    gymnasium==0.29.1 \
    numpy==1.24.3 \
    matplotlib \
    pygame \
    tensorboard

CMD ["bash"]
