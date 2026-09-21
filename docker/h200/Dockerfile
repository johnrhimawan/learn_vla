ARG BASE_IMAGE=nvidia/cuda:13.0.2-runtime-ubuntu24.04
FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM ${BASE_IMAGE}

COPY --from=uv /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    LEROBOT_RUNNER=scripts/run_h200 \
    PATH=/workspace/learn_vla/.venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libegl1 libgl1 libglfw3 python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/learn_vla
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev
COPY . .

CMD ["python", "docker/h200/verify_runtime.py"]
