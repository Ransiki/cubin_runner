#! /usr/bin/env bash

set -eo pipefail

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export CUTLASS_ADDITIONAL_CMAKE_ARGS="-DCUTLASS_ENABLE_INTERNAL_NVVM=ON -DCUTLASS_ENABLE_EXTENDED_PTX=ON ${CUTLASS_ADDITIONAL_CMAKE_ARGS}"

${SHELL} ${THIS_DIR}/build.linux.lsf.sm70.sh "$@"
