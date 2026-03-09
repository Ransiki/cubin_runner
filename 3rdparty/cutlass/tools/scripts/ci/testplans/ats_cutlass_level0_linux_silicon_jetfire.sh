#! /usr/bin/env bash

set -exv

: THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

: CUTLASS_UNIT_TEST_FILTER=${CUTLASS_UNIT_TEST_FILTER:="SM5*:SM6*:SM70*-*batched_gemv*"}

${SHELL} ${THIS_DIR}/ats_cutlass_level0_linux_silicon.sh
