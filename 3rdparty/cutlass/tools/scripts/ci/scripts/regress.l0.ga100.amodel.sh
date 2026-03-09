#! /usr/bin/env bash

set -eo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"

export CUTLASS_UNIT_TEST_FILTER=${CUTLASS_UNIT_TEST_FILTER:="*SM80*"}
export CTEST_TIMEOUT=${CTEST_TIMEOUT:=3600}

MEMCHECK_TOOLS="" bash ${THIS_DIR}/run_on_amodel.sh ${THIS_DIR}/regress.l0.sh
