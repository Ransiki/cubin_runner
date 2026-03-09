#! /usr/bin/env bash

set -eo pipefail

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

echo WORK_DIR=${WORK_DIR:=$(pwd)}
echo CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
echo CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}
echo CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR:=$(pwd)/install}

echo CUTLASS_BUILD_TARGET=${CUTLASS_BUILD_TARGET:=all}
echo CUTLASS_BUILD_TYPE=${CUTLASS_BUILD_TYPE:=Release}
echo CUTLASS_ADDITIONAL_CMAKE_ARGS=${CUTLASS_ADDITIONAL_CMAKE_ARGS:=""}

if [ ! -z ${BUILD_NUMBER+x} ]; then 
  CUTLASS_ADDITIONAL_CMAKE_ARGS="${CUTLASS_ADDITIONAL_CMAKE_ARGS} -DCUTLASS_VERSION_BUILD=${BUILD_NUMBER}"
fi

[ -f "${CUTLASS_SRC_DIR}/CMakeLists.txt" ] || (echo "*** Error: Did not find CMakeLists.txt in CUTLASS_SRC_DIR." && exit 1)

cd ${WORK_DIR}

if [ -n "${CUDA_ARTIFACT}" ]; then
  curl -o cuda.tgz "${CUDA_ARTIFACT}"
  mkdir -p cuda_toolkit && tar -xzf cuda.tgz -C cuda_toolkit
  export CUDA_PATH=$(pwd)/cuda_toolkit
fi

mkdir -p ${CUTLASS_BUILD_DIR} && rm -rf ${CUTLASS_BUILD_DIR}/*
cmake -DCMAKE_BUILD_TYPE=${CUTLASS_BUILD_TYPE} -B${CUTLASS_BUILD_DIR} -H${CUTLASS_SRC_DIR} ${CUTLASS_ADDITIONAL_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CUTLASS_INSTALL_DIR}

cd ${CUTLASS_SRC_DIR}

${THIS_DIR}/make.sh -C "${CUTLASS_BUILD_DIR}" ${CUTLASS_BUILD_TARGET} VERBOSE=1 "${@}"

echo "build.linux.sh done"
