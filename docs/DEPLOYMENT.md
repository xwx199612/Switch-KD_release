# Deployment

Use Linux with an NVIDIA driver and NVIDIA Container Toolkit compatible with CUDA 12.4, and a CUDA GPU supporting BF16. The verified host has four RTX 4000 Ada GPUs with approximately 20 GiB each. The runtime uses `device_map=auto` and rejects CPU/disk placement; an absolute minimum VRAM is not claimed.

The embedded Student is Qwen/Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The embedded adapter SHA256 is recorded in `release_manifest.json` and checksums.
