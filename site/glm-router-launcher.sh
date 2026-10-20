#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]]
/home/spark/.local/libexec/llama-swap/deepseek-v41-1017-vllm --stop
exec /home/spark/.local/libexec/llama-swap/glm-5.3-flash-ringside
