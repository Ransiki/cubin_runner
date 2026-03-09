#! /usr/bin/env bash

set -eo pipefail

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export CUTLASS_ADDITIONAL_CMAKE_ARGS="-DJETFIRE_ENABLED=ON ${CUTLASS_ADDITIONAL_CMAKE_ARGS}"

${SHELL} ${THIS_DIR}/build.linux.lsf.sm70.internal.sh "$@"
