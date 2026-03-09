#! /usr/bin/env bash

set -exvo pipefail

THIS_DIR="$(dirname "$(readlink -f "$0")")"

export CUTLASS_HOST=${CUTLASS_HOST:=$(hostname)}
export CUTLASS_SRC_DIR=${CUTLASS_SRC_DIR:=$(pwd)}
export CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:=$(pwd)/build}
export CUTLASS_INSTALL_DIR=${CUTLASS_INSTALL_DIR:=$(pwd)/install}

echo EXECUTOR_NUMBER=${EXECUTOR_NUMBER:=0}
echo BUILD_TAG=${BUILD_TAG:=jenkins}

echo WORK_DIR=${WORK_DIR:=/tmp/cutlass-${CUTLASS_HOST}-${EXECUTOR_NUMBER}-$(echo -n ${BUILD_TAG} | md5sum | cut -d ' ' -f 1)}
echo SRC_DIR=${SRC_DIR:=${WORK_DIR}/src}
echo BUILD_DIR=${BUILD_DIR:=${WORK_DIR}/build}
echo INSTALL_DIR=${INSTALL_DIR:=${WORK_DIR}/install}

mkdir -p ${WORK_DIR} && pushd ${WORK_DIR}

checked_rsync_from_there() {
  echo "rsync ${CUTLASS_HOST}:${1} to ${2}"
  ssh -o StrictHostKeyChecking=no -q ${CUTLASS_HOST} "[ -d \"${1}\" ]" || return 0
  mkdir -p ${2}
  rsync --stats -a -e "ssh -o StrictHostKeyChecking=no -q" -q ${CUTLASS_HOST}:${1}/ ${2}/
}

checked_rsync_to_there() {
  [ -d ${1} ] || return 0
  echo "rsync ${1} to ${CUTLASS_HOST}:${2}"
  ssh -o StrictHostKeyChecking=no -q ${CUTLASS_HOST} "mkdir -p \"${2}\"" || return 0
  rsync --stats -a -e "ssh -o StrictHostKeyChecking=no -q" -q ${1}/ ${CUTLASS_HOST}:${2}/
}

rm -rf ${SRC_DIR} ${BUILD_DIR} ${INSTALL_DIR} && mkdir -p ${SRC_DIR} ${BUILD_DIR}

# The following sub-shell will allow us to copy the working directory to and from a
# local scratch space. Once we start building, we want to copy the build output back to
# the original host machine even if the build fails, so we need to capture the error
# to return at the end but make sure we copy the files back first, and then let the 
# scratch space get deleted before returning so we don't fill up the /tmp space on the
# this (remote) LSF agent.

ERROR=0
(
  set -exvo pipefail
  ERROR=0
  mkdir -p ${SRC_DIR} || exit $?
  checked_rsync_from_there ${CUTLASS_SRC_DIR} ${SRC_DIR} || exit $?
  checked_rsync_from_there ${CUTLASS_BUILD_DIR} ${BUILD_DIR} || exit $?
  checked_rsync_from_there ${CUTLASS_INSTALL_DIR} ${INSTALL_DIR} || exit $?
  cd ${SRC_DIR} && WORK_DIR=${WORK_DIR} CUTLASS_SRC_DIR=${SRC_DIR} CUTLASS_BUILD_DIR=${BUILD_DIR} CUTLASS_INSTALL_DIR=${INSTALL_DIR} ${@} || ERROR=$?
  checked_rsync_to_there ${BUILD_DIR} ${CUTLASS_BUILD_DIR} || ERROR=$?
  checked_rsync_to_there ${INSTALL_DIR} ${CUTLASS_INSTALL_DIR} || ERROR=$?
  exit ${ERROR}
) || ERROR=$?

popd && rm -rf ${WORK_DIR}

exit ${ERROR}
