#! /usr/bin/env bash

set -eo pipefail

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export CUTLASS_HOST=${CUTLASS_HOST:=$(hostname)}
export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}
export CUTLASS_BUILD_TARGET=${CUTLASS_BUILD_TARGET:=all}
export CUTLASS_BUILD_TYPE=${CUTLASS_BUILD_TYPE:=Release}
export CUTLASS_ADDITIONAL_CMAKE_ARGS=${CUTLASS_ADDITIONAL_CMAKE_ARGS:=""}

echo EXECUTOR_NUMBER=${EXECUTOR_NUMBER:=0}
echo BUILD_TAG=${BUILD_TAG:=jenkins}

echo WORK_DIR=${WORK_DIR:=/tmp/cutlass-${CUTLASS_HOST}-${EXECUTOR_NUMBER}-${BUILD_TAG}}
echo SRC_DIR=${SRC_DIR:=${WORK_DIR}/src}
echo BUILD_DIR=${BUILD_DIR:=${WORK_DIR}/build}

ERROR=0
(
  set -eo pipefail
  mkdir -p ${SRC_DIR}
  rsync -a -e "ssh -o StrictHostKeyChecking=no" ${CUTLASS_HOST}:${CUTLASS_SRC_DIR}/ ${SRC_DIR}/
  CUTLASS_SRC_DIR=${SRC_DIR} CUTLASS_BUILD_DIR=${BUILD_DIR} bash ${SRC_DIR}/tools/scripts/ci/scripts/build.linux.sh
  rsync -a -e "ssh -o StrictHostKeyChecking=no" ${BUILD_DIR}/ ${CUTLASS_HOST}:${CUTLASS_BUILD_DIR}/
) || ERROR=$?

rm -rf ${WORK_DIR}

exit ${ERROR}
