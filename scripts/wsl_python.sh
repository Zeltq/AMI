#!/usr/bin/env bash
set -euo pipefail
export HSA_ENABLE_DXG_DETECTION=1
export ROCPROFILER_REGISTER_ENABLED=0
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/mnt/c/AMI/src${PYTHONPATH:+:$PYTHONPATH}
exec "$HOME/dota-compat-check/bin/python" -u "$@"
