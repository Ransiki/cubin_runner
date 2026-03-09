#! /usr/bin/env bash

set -eo pipefail
ERROR=0

: THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}
echo CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:="$(pwd)/build"}
echo CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR:="${CUTLASS_BUILD_DIR}/install"}
echo CTEST_TIMEOUT=${CTEST_TIMEOUT:=600}

[ -z ${CUTLASS_UNIT_TEST_FILTER+x} ] || GTEST_FILTER_ARGS="--gtest_filter=${CUTLASS_UNIT_TEST_FILTER}"

export CUTLASS_ARTIFACT_DIR=${CUTLASS_ARTIFACT_DIR:=$(pwd)/artifacts}
export CUTLASS_EXTERNAL_DIR=${CUTLASS_EXTERNAL_DIR:=$(pwd)/external}

: LOAD_EXE_BEFORE_EXECUTION=${LOAD_EXE_BEFORE_EXECUTION:=0}

if [ ! -z "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]; then
  # Jenkins may override the allowed devices on the machine.
  echo "Overriding CUDA_VISIBLE_DEVICES with ${CUDA_VISIBLE_DEVICES_OVERRIDE}"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE}
fi

if [ -n "${CUDA_ARTIFACT+x}" ]; then
  
  curl -o cuda.tgz "${CUDA_ARTIFACT}"
  mkdir -p cuda_toolkit && tar -xzf cuda.tgz -C cuda_toolkit
  CUDA_PATH=$(pwd)/cuda_toolkit
  
elif [ -n "${CUDA_PATH+x}" ]; then

  : # Leave CUDA_PATH alone.

elif [ -n "${CUDA_DIR+x}" ]; then

  CUDA_PATH=${CUDA_DIR}

elif NVCC_FOUND=$(command -v nvcc); then

  CUDA_PATH=$(readlink -nm ${NVCC_FOUND}/../..)

else

  # If CUDA_DIR is not specified, pull it from Artifactory. We plan to include CUDA
  # as a proper CMake component soon, then this workaround can go away.

  echo CUDA_VERSION=${CUDA_VERSION:=10.2.123}
  CUDA_VERSION_MM=$(echo ${CUDA_VERSION} | sed -nr 's/([0-9]+\.[0-9]+)(\.[0-9]+)?/\1/p')
  echo CUDA_FLAVOR=${CUDA_FLAVOR:=release-internal}
  echo CUDA_REVISION=${CUDA_REVISION:=27847785}

  echo CUDA_ARTIFACT_FILE=${CUDA_ARTIFACT_FILE:=cuda-${CUDA_VERSION}-${CUDA_REVISION}.tgz}
  echo CUDA_ARTIFACT=${CUDA_ARTIFACT:=${CUTLASS_ARTIFACT_DIR}/cuda/${CUDA_VERSION_MM}/${CUDA_FLAVOR}/${CUDA_ARTIFACT_FILE}}
  echo CUDA_ARTIFACT_URL=${CUDA_ARTIFACT_URL:=https://sc-hw-artf.nvidia.com/artifactory/compute-generic-local/cuda/${CUDA_VERSION_MM}/x86_64/linux/generic/${CUDA_FLAVOR}/${CUDA_ARTIFACT_FILE}}

  if [ ! -f "${CUDA_ARTIFACT}" ]; then
    CUDA_ARTIFACT_DIR=$(dirname ${CUDA_ARTIFACT})
    mkdir -p ${CUDA_ARTIFACT_DIR}
    wget -nv ${CUDA_ARTIFACT_URL} -O ${CUDA_ARTIFACT}.tmp
    # Survive CTRL-C with temporary intermediate file
    mv ${CUDA_ARTIFACT}.tmp ${CUDA_ARTIFACT}
  fi

  echo CUDA_EXTERNAL_DIR=${CUDA_EXTERNAL_DIR:=${CUTLASS_EXTERNAL_DIR}/cuda/${CUDA_FLAVOR}/${CUDA_VERSION}-${CUDA_REVISION}}

  if [ ! -f ${CUDA_EXTERNAL_DIR}/bin/cuda-memcheck ]; then
    mkdir -p ${CUDA_EXTERNAL_DIR}
    tar -xzf ${CUDA_ARTIFACT} -C ${CUDA_EXTERNAL_DIR}
  fi

  CUDA_PATH=${CUDA_EXTERNAL_DIR}
fi

export PATH=${CUTLASS_INSTALL_DIR}/test/cutlass/bin:${CUTLASS_INSTALL_DIR}/bin:${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUTLASS_INSTALL_DIR}/test/cutlass/lib64:${CUTLASS_INSTALL_DIR}/lib64:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}

if [ -f "${CUDA_PATH}/Sanitizer/compute-sanitizer" ]; then
  CUTLASS_TEST_EXECUTION_ENVIRONMENT_DEFAULT=${CUDA_PATH}/Sanitizer/compute-sanitizer
elif [ -f "${CUDA_PATH}/compute-sanitizer/compute-sanitizer" ]; then
  CUTLASS_TEST_EXECUTION_ENVIRONMENT_DEFAULT=${CUDA_PATH}/compute-sanitizer/compute-sanitizer
else
  CUTLASS_TEST_EXECUTION_ENVIRONMENT_DEFAULT=${CUDA_PATH}/bin/cuda-memcheck
fi

echo CUTLASS_TEST_EXECUTION_ENVIRONMENT=${CUTLASS_TEST_EXECUTION_ENVIRONMENT:=${CUTLASS_TEST_EXECUTION_ENVIRONMENT_DEFAULT}}
# : MEMCHECK_TOOLS=${MEMCHECK_TOOLS:=memcheck racecheck synccheck initcheck}
# echo CUTLASS_TEST_EXECUTION_SANITIZER_TOOLS=${CUTLASS_TEST_EXECUTION_SANITIZER_TOOLS:=${MEMCHECK_TOOLS}}

# nvidia-smi -L

if [ ! -z "${CUDA_VISIBLE_DEVICES}" ] && [ ! -z "${EXECUTOR_NUMBER}" ]; then
  # Run the regressions on a single device spread across all allowed matching devices
  # on the agent.
  IFS=',' read -ra CVDA <<< ${CUDA_VISIBLE_DEVICES}
  export CUDA_VISIBLE_DEVICES=${CVDA[$(expr ${EXECUTOR_NUMBER} % ${#CVDA[@]})]}
  echo "Using CUDA device ${CUDA_VISIBLE_DEVICES}."
fi

echo "Running unit tests ..."

if [ -z ${LSB_HOSTS+x} ]; then
  : ${CTEST_J:=$(grep -c ^processor /proc/cpuinfo)}
  : ${CTEST_EXE:=ctest}
else
  : ${CTEST_J:=$(echo ${LSB_HOSTS} | wc -w)}
  : ${CTEST_EXE:=/home/utils/cmake-3.12.4/bin/ctest}
fi

(
  set -e
  ERROR=0
  # if [ ${LOAD_EXE_BEFORE_EXECUTION} -ne 0 ]; then
  #   ldd "${CUTLASS_UNIT_TEST_EXE}" > /dev/null
  # fi
  cd ${CUTLASS_INSTALL_DIR}/test/cutlass
  CUTLASS_TEST_EXECUTION_ENVIRONMENT="${CUTLASS_TEST_EXECUTION_ENVIRONMENT}" ${CTEST_EXE} -T Test --output-on-failure --no-compress-output --test-output-size-passed 1000000 --test-output-size-failed 1000000 --timeout ${CTEST_TIMEOUT} -j ${CTEST_J} ${CTEST_ARGS}
) || ERROR=$?

echo "... done with unit tests."

[ ${ERROR} -ne 0 ] && echo "*** Failures detected!" >&2 

exit ${ERROR}
