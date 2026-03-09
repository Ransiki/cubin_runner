#! /usr/bin/env bash

set -exvo pipefail

export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)}
export MEMCHECK_TOOLS=${MEMCHECK_TOOLS:=""}

./tools/scripts/ci/scripts/regress.l0.sh
