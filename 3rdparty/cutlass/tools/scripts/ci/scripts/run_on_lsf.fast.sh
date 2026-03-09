#! /usr/bin/env bash

set -eo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"

export CUTLASS_HOST=${CUTLASS_HOST:=$(hostname)}
export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}

# export NODE_NAME=${NODE_NAME:=${CUTLASS_HOST}}
export EXECUTOR_NUMBER=${EXECUTOR_NUMBER:=$$}
export BUILD_TAG=${BUILD_TAG:=jenkins}

: LSF_CORES_MIN=${LSF_CORES_MIN:=8}
: LSF_CORES_MAX=${LSF_CORES_MAX:=8}

/home/nv/bin/nvprojectname save . business=gpu chip=g000 group=arch team=compute subteam=fastkernels

TRAMPOLINE=/tmp/cutlass-${CUTLASS_HOST}-${EXECUTOR_NUMBER}-${BUILD_TAG}-run.sh

qsub -q o_cpu_32G_8H \
  -app build \
  -R 'span[hosts=1]' \
  -R 'rusage[mem=16000]' \
  -R 'rusage[tmp=4000]' \
  -R 'select[tmp>16000]' \
  -n ${LSF_CORES_MIN},${LSF_CORES_MAX} \
  -E "rsync -e \"ssh -o StrictHostKeyChecking=no -q\" ${CUTLASS_HOST}:${THIS_DIR}/run_on_lsf.fast.callback.sh ${TRAMPOLINE}" \
  -Ep "rm -f ${TRAMPOLINE}" \
  -I "${TRAMPOLINE} ${@}"
