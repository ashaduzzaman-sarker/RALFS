# Docker Deployment Guide for RALFS

This guide explains how to build and run RALFS using Docker.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (optional, for easier management)
- For GPU support: NVIDIA Docker runtime

## Quick Start

### Building the Docker Image

```bash
docker build -t ralfs:latest .
```

### Running with Docker

**Show help:**
```bash
docker run --rm ralfs:latest --help
```

**Preprocess data:**
```bash
docker run --rm -v $(pwd)/data:/app/data ralfs:latest preprocess --dataset arxiv --max-samples 100
```

**Build indexes:**
```bash
docker run --rm -v $(pwd)/data:/app/data ralfs:latest build-index --dataset arxiv
```

**Run complete pipeline:**
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ralfs:latest pipeline --dataset arxiv --max-samples 10
```

### Using Docker Compose

Docker Compose simplifies volume mounting and configuration.

**Run any command:**
```bash
docker-compose run --rm ralfs preprocess --dataset arxiv --max-samples 100
```

**Run pipeline:**
```bash
docker-compose run --rm ralfs pipeline --dataset arxiv --max-samples 10
```

## GPU Support

To enable GPU support, you need to install the NVIDIA Container Toolkit and uncomment the GPU-related lines in `docker-compose.yml`.

### Install NVIDIA Container Toolkit

```bash
# For Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Run with GPU

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ralfs:latest train --dataset arxiv
```

Or with Docker Compose (after uncommenting GPU sections):
```bash
docker-compose run --rm ralfs train --dataset arxiv
```

## Volume Mounts

The following directories should be mounted as volumes for data persistence:

- `/app/data` - Dataset storage
- `/app/results` - Evaluation results and generated summaries
- `/app/outputs` - Temporary outputs
- `/app/checkpoints` - Model checkpoints
- `/app/configs` - Configuration files (optional)

## Environment Variables

You can pass environment variables for API keys and configurations:

```bash
docker run --rm \
  -e WANDB_API_KEY=your_key \
  -e HF_TOKEN=your_token \
  -v $(pwd)/data:/app/data \
  ralfs:latest train --dataset arxiv
```

Or use an `.env` file with Docker Compose:

```bash
# .env file
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
```

## Building for Production

For production deployments, consider:

1. **Multi-stage builds** to reduce image size
2. **Specific base images** with GPU support if needed
3. **Caching layers** for faster rebuilds

### Example Multi-stage Build

```dockerfile
# Build stage
FROM python:3.10-slim as builder
# ... install dependencies ...

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /app /app
# ... minimal runtime dependencies ...
```

## Troubleshooting

### Out of Memory

If you encounter memory issues, increase Docker's memory limit:

```bash
# For Docker Desktop, go to Settings > Resources > Memory
# For command line:
docker run --memory=8g --rm ralfs:latest ...
```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run with the same user ID as your host user
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data \
  ralfs:latest ...
```

### ColBERT Installation Failures

ColBERT installation may fail in some environments. This is optional and won't prevent the basic functionality from working.

## Best Practices

1. **Use specific tags** for production: `ralfs:v1.0.0` instead of `latest`
2. **Mount volumes** for all data directories to preserve work
3. **Use .dockerignore** to exclude unnecessary files from the build context
4. **Set resource limits** to prevent container from consuming all system resources
5. **Use environment variables** for sensitive configuration

## Examples

### Complete Training Workflow

```bash
# 1. Preprocess
docker-compose run --rm ralfs preprocess --dataset arxiv --max-samples 1000

# 2. Build indexes
docker-compose run --rm ralfs build-index --dataset arxiv

# 3. Train
docker-compose run --rm ralfs train --dataset arxiv --config configs/train/default.yaml

# 4. Generate summaries
docker-compose run --rm ralfs generate \
  input.jsonl \
  --checkpoint checkpoints/best_model \
  --output results/summaries.json

# 5. Evaluate
docker-compose run --rm ralfs evaluate \
  results/summaries.json \
  data/references.json \
  --metrics rouge,bertscore,egf
```

### Development Mode

For development, you can mount the source code and run tests:

```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ralfs:latest \
  poetry run pytest tests/ -v
```
