# DeepGait runs a Node/Express API that shells out to a Python process for
# pose extraction + PyTorch inference, plus a React SPA served by that same
# API. Render's native "Node" and "Python" runtimes each only ship one
# language, so this image builds both into one container.

FROM node:20-bookworm

# python3-dev + build-essential cover packages that need to compile;
# libglib2.0-0/libgl1 satisfy opencv-python-headless's runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node deps first so this layer is cached unless package*.json change.
COPY package.json package-lock.json* ./
COPY server/package.json server/package.json
COPY client/package.json client/package.json
RUN npm install

# Install Python deps next, cached unless pyproject.toml or server/ml change.
# (setuptools' editable install resolves "ml" -> server/ml, so that source
# tree must exist before `pip install -e` runs.) Torch is installed from the
# CPU wheel index first so the default install doesn't pull ~2GB of CUDA
# libraries that are useless without an NVIDIA GPU (Render web services are
# CPU-only unless on a GPU-specific plan).
COPY pyproject.toml ./
COPY server/ml server/ml
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && /app/.venv/bin/pip install --no-cache-dir -e ".[inference]"

# Now bring in the rest of the source and build the client bundle.
COPY . .
RUN npm run build:client

# Bake the YOLO11-pose weights into the image (at the exact path pose.py
# expects) so the first request doesn't pay a cold-start download.
RUN /app/.venv/bin/python -c "from ultralytics import YOLO; YOLO('/app/server/ml/weights/yolo11n-pose.pt')" || true

ENV PYTHON_BIN=/app/.venv/bin/python
ENV NODE_ENV=production

EXPOSE 3001

CMD ["npm", "run", "start", "-w", "deepgait-server"]
