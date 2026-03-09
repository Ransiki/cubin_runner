#! /usr/bin/env bash

: THIS_DIR=${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

: LSF_CORES_MIN=${LSF_CORES_MIN:=1}
: LSF_CORES_MAX=${LSF_CORES_MAX:=4}

bash ${THIS_DIR}/run_on_lsf.fast.sh ./tools/scripts/ci/scripts/regress.l0.ga100.amodel.sh
