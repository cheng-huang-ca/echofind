#!/bin/bash
# Setup script for the `echofind` Claude Code cloud environment (paste into the environment's
# Setup script box). It only installs packages: it must finish in about 5 minutes to be cached,
# so data downloads stay in `make data`. See docs/cloud-setup.md.
set -euo pipefail
uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --system numpy scipy xarray pyyaml pandas pyarrow matplotlib scikit-learn \
  lightgbm shap mlflow echopype timm onnx onnxruntime pybind11 pytest ruff
