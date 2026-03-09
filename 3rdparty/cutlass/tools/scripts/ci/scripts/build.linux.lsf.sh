#! /usr/bin/env bash

set -eo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"

export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}
export CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR:=$(pwd)/install}
export CUTLASS_BUILD_TARGET=${CUTLASS_BUILD_TARGET:=all}
export CUTLASS_BUILD_TYPE=${CUTLASS_BUILD_TYPE:=Release}
export CUTLASS_ADDITIONAL_CMAKE_ARGS=${CUTLASS_ADDITIONAL_CMAKE_ARGS:=""}

: LSF_CORES_MIN=${LSF_CORES_MIN:=8}
: LSF_CORES_MAX=${LSF_CORES_MAX:=8}

/home/nv/bin/nvprojectname save . business=gpu chip=g000 group=arch team=compute subteam=fastkernels

qsub -I -q o_cpu_16G_8H \
  -R 'span[hosts=1]' -R 'affinity[core(1):membind=localprefer:distribute=pack]' \
  -R 'select[defined(rel68)]' \
  -n ${LSF_CORES_MIN},${LSF_CORES_MAX} \
  ${THIS_DIR}/build.linux.sh
