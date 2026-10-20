#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then set -- --execute; fi
exec /home/spark/ai/spark-ds41f/site/managed-launcher.sh "$@"
