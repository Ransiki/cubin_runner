#! /usr/bin/env bash

set -eo pipefail


check_dependencies () {
    
    required_environment_vars=(
        CUDA_PATH
        CUTLASS_SRC_DIR
        CUTLASS_BUILD_DIR
        CUTLASS_INSTALL_DIR
        TEST_LEVEL
    )
 
    for var in ${required_environment_vars[@]}; do
        if [ -z ${!var+x} ]; then 
            echo "Please set environment variable: ${var}"
            exit 1
        else
            echo "${var} is set to ${!var}"
        fi
    done

    if [ $TEST_LEVEL != "L0" -a $TEST_LEVEL != "L1" ]; then 
        echo "Unknown TEST_LEVEL: ${TEST_LEVEL}"
        exit 1
    fi
}

check_dependencies

SM_VERSION=80

THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

if [ $TEST_LEVEL = "L0" ]; then 
    KERNELS=$(paste -sd, ${THIS_DIR}/L0_kernels.txt)
else
    KERNELS="all"
fi

export GCC_DIR=${GCC_DIR:="/home/utils/gcc-5.5.0"}
export CMAKE_DIR=${CMAKE_DIR:="/home/utils/cmake-3.12.4"}
export PYTHON3_DIR=${PYTHON3_DIR:="/home/utils/Python-3.7.3"}

export PATH="${GCC_DIR}/bin:${CMAKE_DIR}/bin:${CUDA_PATH}/bin:${PYTHON3_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${GCC_DIR}/lib64:${CMAKE_DIR}/lib64:${CUDA_PATH}/lib64:$LD_LIBRARY_PATH"

export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR}
export CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR}
export CUTLASS_ADDITIONAL_CMAKE_ARGS="-DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_OPERATIONS=gemm -DCUTLASS_LIBRARY_KERNELS=$KERNELS -DCUTLASS_ENABLE_EXTENDED_PTX=ON"


CMD="${SHELL} ${THIS_DIR}/../build.linux.lsf.fast.sh install"

echo "Executing: ${CMD}"

${CMD}

exit
