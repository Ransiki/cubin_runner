#! /usr/bin/env bash

set -eo pipefail

: ${THIS_DIR:="$(dirname "$(readlink -f "$0")")"}

if [ -z ${LSB_HOSTS+x} ]; then
  : ${MAKE_J:=$(grep -c ^processor /proc/cpuinfo)}
  : ${MAKE_EXE:=make}
else
  : ${MAKE_J:=$(echo ${LSB_HOSTS} | wc -w)}
  : ${MAKE_EXE:=/home/utils/make-3.82/bin/make}
fi

echo "[Make] Utilizing ${MAKE_J} cores"

${MAKE_EXE} --no-print-directory -j ${MAKE_J} "${@}"

echo "make.sh done."
