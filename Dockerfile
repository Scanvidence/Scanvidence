# Matches the reproducibility commitment in Section 5.1 / 9.1 of the proposal:
# a Docker image with pinned versions, released alongside results.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies before copying source so this layer caches across
# ordinary code changes — only rebuilds when pyproject.toml changes.
COPY pyproject.toml README.md ./
COPY scanvidence/ scanvidence/
RUN pip install --no-cache-dir -e ".[segmentation,tracking]"

COPY . .

# Determinism controls from Section 5.1 — set at runtime, not baked into
# the image, so a single image can run both deterministic and fast modes.
ENV PYTHONHASHSEED=0

CMD ["python", "-m", "scanvidence"]
