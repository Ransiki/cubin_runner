#! /usr/bin/env bash

set -eo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"

export CUTLASS_HOST=${CUTLASS_HOST:=$(hostname)}
export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}
export CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR:=$(pwd)/install}

# export NODE_NAME=${NODE_NAME:=${CUTLASS_HOST}}
export EXECUTOR_NUMBER=${EXECUTOR_NUMBER:=0}
export BUILD_TAG=${BUILD_TAG:=jenkins}

rm -rf ${CUTLASS_BUILD_DIR}

${SHELL} ${THIS_DIR}/run_on_lsf.fast.sh ./tools/scripts/ci/scripts/build.linux.sh "${@}"
