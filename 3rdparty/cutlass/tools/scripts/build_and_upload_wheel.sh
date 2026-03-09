#!/bin/bash

# {$nv-internal-release file}

# Script for building a Python wheel for the CUTLASS Python interface, cutlass_library, and pycute.
# The associated wheel is uploaded to artifactory 

MINARG=2
MAXARG=3
if [ $# -lt $MINARG ] || [ $# -gt $MAXARG ]; then
  echo "Usage: $0 <cutlass_root> <version> [<upload>]"
  echo "  cutlass_root: path to root of CUTLASS directory"
  echo "  version: version tag of the wheel. Must match that in pyproject.toml"
  echo "  upload (optional): 0 or 1. If 1, the wheel is uploaded to artifactory for publishing"
  exit 1
fi

# Raise an error if artifactory username and token are not available
if [[ -z "${URM_USER}" ]]; then
  echo "Environment variable URM_USER must be set to the username associated with the account used to access urm.nvidia.com"
  exit 1
fi

if [[ -z "${URM_TOKEN}" ]]; then
  echo "Environment variable URM_TOKEN must be set to an access token associated with the account used to access urm.nvidia.com"
  exit 1
fi

cutlass_root=$1
version_no=$2

if [ $# -eq $MAXARG ]; then
  upload=$3
else
  upload=0
fi

curdir=$(pwd)
cd $cutlass_root

# Verify that the directory has been stripped of any internal IP. Wheels should
# only be generated on publicly-released code.
if [ -f tools/scripts/guardwords_public.sh ]; then
  RED='\033[0;31m'
  NC='\033[0m'
  echo -e "${RED}ERROR: Attempting to generate and upload a wheel using an internal verson of CUTLASS."
  echo -e "Wheels should only be generated using publicly-release code.${NC}"
  cd $curdir
  exit 1
fi

# Build the wheel
python -m build

# Upload the wheel to artifactory with the required annotations
if [ $upload -eq 1 ]; then
  component_name="component_name=nvidia-cutlass"
  arch="arch=x86_64"
  os="os=linux"
  version="version=${version_no}"
  branch="branch=${version_no}"
  release_approver="release_approver=jkosaian"
  release_status="release_status=ready"

  # Skip security checks related to benign uses of subprocess
  skip_rules="wheeltamer_rule_skip=B404,B603,B605"

  curl --user "${URM_USER}:${URM_TOKEN}" \
      --upload-file dist/nvidia_cutlass-${version_no}-py3-none-any.whl \
      --request PUT "https://urm.nvidia.com/artifactory/sw-dl-cask-pypi-local/nvidia-cutlass/${version_no}/${version_no}/nvidia_cutlass-${version_no}-py3-none-any.whl;${component_name};${os};${arch};${version};${branch};${release_approver};${release_status};${skip_rules}"
 fi

cd $curdir
