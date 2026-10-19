# GPU allocation status

The previous allocation request is resolved. This instance exposes four NVIDIA L40 GPUs and CUDA works in `slm-lab/.venv`. See [hardware report](reports/hardware.json) and [four-GPU readiness run](checkpoints/l40-ddp-readiness/training.json).

The project did not rent or attach external GPUs; the platform exposes the current allocation. The prior host advertised A100s without usable device access. That historical inventory is not the current training hardware.
