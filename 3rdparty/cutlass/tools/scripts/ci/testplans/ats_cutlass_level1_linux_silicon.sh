#! /usr/bin/env bash

set -exvo pipefail

echo THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

export MEMCHECK_TOOLS=${MEMCHECK_TOOLS:="memcheck racecheck initcheck"}

${THIS_DIR}/ats_cutlass_level0_linux_silicon.sh
