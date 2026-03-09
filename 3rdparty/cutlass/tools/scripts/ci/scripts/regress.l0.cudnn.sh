#! /usr/bin/env bash

set -eo pipefail
ERROR=0

: THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:="$(pwd)/build"}
export CUTLASS_ARTIFACT_DIR=${CUTLASS_ARTIFACT_DIR:=$(pwd)/artifacts}
export CUTLASS_EXTERNAL_DIR=${CUTLASS_EXTERNAL_DIR:=$(pwd)/external}

export MEMCHECK_TOOLS=${MEMCHECK_TOOLS=""}

if [ -n "${CUDNN_PATH+x}" ]; then

  : # Leave CUDNN_PATH alone.

#? elif CUDA_FOUND=$(command -v nvcc); then
#? 
#?   CUDA_PATH=$(readlink -nm ${NVCC_FOUND}/../..)
#? 
else

  # If CUDNN_PATH is not specified, pull it from Artifactory.
    
  echo CUDNN_VERSION=${CUDNN_VERSION:=7.4.0}
  CUDNN_VERSION_MM=$(echo ${CUDNN_VERSION} | sed -nr 's/([0-9]+\.[0-9]+)(\.[0-9]+)?/\1/p')
  echo CUDNN_FLAVOR=${CUDNN_FLAVOR:=release}
  echo CUDNN_REVISION=${CUDNN_REVISION:=26053803}

  echo CUDNN_ARTIFACT_FILE=${CUDNN_ARTIFACT_FILE:=cudnn-${CUDNN_VERSION}-${CUDNN_REVISION}.tgz}
  echo CUDNN_ARTIFACT=${CUDNN_ARTIFACT:=${CUTLASS_ARTIFACT_DIR}/cudnn/${CUDNN_VERSION_MM}/${CUDNN_FLAVOR}/${CUDNN_ARTIFACT_FILE}}
  echo CUDNN_ARTIFACT_URL=${CUDNN_ARTIFACT_URL:=https://sc-hw-artf.nvidia.com/artifactory/compute-generic-local/cudnn/${CUDNN_VERSION_MM}/x86_64/linux/generic/${CUDNN_FLAVOR}/${CUDNN_ARTIFACT_FILE}}

  if [ ! -f "${CUDNN_ARTIFACT}" ]; then
    CUDNN_ARTIFACT_DIR=$(dirname ${CUDNN_ARTIFACT})
    mkdir -p ${CUDNN_ARTIFACT_DIR}
    wget -nv ${CUDNN_ARTIFACT_URL} -O ${CUDNN_ARTIFACT}.tmp
    # Survive CTRL-C with temporary intermediate file
    mv ${CUDNN_ARTIFACT}.tmp ${CUDNN_ARTIFACT}
  fi

  echo CUDNN_EXTERNAL_DIR=${CUDNN_EXTERNAL_DIR:=${CUTLASS_EXTERNAL_DIR}/cuda/${CUDNN_FLAVOR}/${CUDNN_VERSION}-${CUDNN_REVISION}}

  if [ ! -f ${CUDNN_EXTERNAL_DIR}/bin/cuda-memcheck ]; then
    mkdir -p ${CUDNN_EXTERNAL_DIR}
    tar -xvf ${CUDNN_ARTIFACT} -C ${CUDNN_EXTERNAL_DIR}
  fi

  CUDNN_PATH=${CUDNN_EXTERNAL_DIR}
  
fi

export LD_LIBRARY_PATH=${CUDNN_PATH}/lib64:${LD_LIBRARY_PATH}

${SHELL} ${THIS_DIR}/regress.l0.sh "$@"
