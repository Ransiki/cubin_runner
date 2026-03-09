#! /usr/bin/env bash

set -exv

unset CUDA_VERSION
# Need to unset the version so we'll pull one automatically 
# as part of the regress script.

CUTLASS_SRC_DIR=$(pwd) CUTLASS_BUILD_DIR=$(pwd) ./tools/scripts/ci/scripts/regress.l0.ga100.amodel.sh
