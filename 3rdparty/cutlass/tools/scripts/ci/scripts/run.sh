#!/bin/bash

# DEPRECATED: USE regress.lo.sh instead.

set -e

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:="$(pwd)"}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:="$(pwd)"}

export CUDA_PATH=${CUDA_PATH:="$(readlink -f $(dirname $(command -v nvcc))/..)"}
[ -e ${CUDA_PATH}/bin/cuda-memcheck ] || (echo "*** Error: CUDA_PATH invalid and could not be automatically determined!" >&2 && exit 1)

${THIS_DIR}/regress.l0.sh
