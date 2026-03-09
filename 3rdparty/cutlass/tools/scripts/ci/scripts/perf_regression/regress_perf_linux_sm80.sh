#! /usr/bin/env bash

set -eo pipefail

check_dependencies () {
    
    required_environment_vars=(
        CUDA_PATH
        CUTLASS_INSTALL_DIR
        GIT_CL
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

## ---------- Copy build dirs to perf regression scratch space ----------- ##

SCRATCH_DIR="/home/scratch.fast_kernels_perf_regress/cutlass/level_${TEST_LEVEL}"
TIMESTAMP=$(date +%m-%d-%y_%H-%M)
RUN_DIR_TAG="${GIT_CL}_${TIMESTAMP}"
RUN_DIR="${SCRATCH_DIR}/${RUN_DIR_TAG}"

if [ -d $RUN_DIR ]; then
    echo "Dir. already exists: ${RUN_DIR}"
    exit 1
else
    mkdir -p ${RUN_DIR}
fi


echo "Copying install directory to ${RUN_DIR}"
echo "Time: $(date +%H:%M\ \%m/%d/%Y)"

SCRATCH_INSTALL_DIR="${RUN_DIR}/install"
SCRATCH_RESULTS_DIR="${RUN_DIR}/perf_regression_output"

CMD="cp -r ${CUTLASS_INSTALL_DIR} ${SCRATCH_INSTALL_DIR}"
echo -e "Executing: \n${CMD}"
${CMD}
echo -e "Done.\n\n"

## --------------------- Launch tests on computelab ------------------------ ##
export CUDA_PATH=${CUDA_PATH}
export PYTHON3_DIR=${PYTHON3_DIR:="/home/utils/Python-3.7.3"}

export PATH="${CUDA_PATH}/bin:${PYTHON3_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:$LD_LIBRARY_PATH"

export CASK_TESTER_PATH="${SCRATCH_INSTALL_DIR}/bin/cutlass-gemm-tester.exe"
export OUTPUT_DIR=${SCRATCH_RESULTS_DIR}
export SRC="cutlass"
export OP="gemm"
export LEVEL=${TEST_LEVEL}
export GIT_CL=${GIT_CL}

echo "Launching tests ..."
echo "Time: $(date +%H:%M\ \%m/%d/%Y)"

CMD="${SHELL} ${SCRATCH_INSTALL_DIR}/lib64/Python3/site-packages/trace_generation_library/perf_regression/scripts/run_tests_local.sh"

echo -e "Executing: \n${CMD}"
${CMD}
echo -e "Done.\nTime: $(date +%H:%M\ \%m/%d/%Y)"

exit
