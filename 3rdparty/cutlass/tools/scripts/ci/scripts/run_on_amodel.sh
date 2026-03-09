#! /usr/bin/env bash

set -eo pipefail

if [ -n "${VERBOSE+x}" ]; then
  set -x
fi

#----------------------------------------------------------
# Parameter Validation
#----------------------------------------------------------

# Check if required arguments are provided
if [ $# -lt 2 ]; then
  echo "Error: This script requires at least 2 arguments:"
  echo "  Usage: $0 <cuda_sm> <repo_component> [additional_args...]"
  echo "  Example: $0 sm107 cutlass"
  echo ""
  echo "Arguments:"
  echo "  cuda_sm: CUDA SM Version (e.g., sm100, sm103, sm107)"
  echo "  repo_component: component to test (cutlass for cutlass/cask, dkg_ir for DKG *_ir components)"
  echo ""
  echo "Additional arguments will be passed to the test execution command."
  exit 1
fi

# Extract required arguments
CUDA_SM="$1"
REPO_COMPONENT="$2"
shift 2  # Remove the first two arguments, leaving additional args in $@

# Validate CUDA_SM format (should start with 'sm' followed by numbers)
if [[ ! "$CUDA_SM" =~ ^sm[0-9]+$ ]]; then
  echo "Error: Invalid CUDA_SM format: '$CUDA_SM'"
  echo "  CUDA_SM should be in format 'sm<number>' (e.g., sm100, sm103, sm107)"
  exit 1
fi

# Validate REPO_COMPONENT (only cutlass and dkg_ir are supported)
if [[ "$REPO_COMPONENT" != "cutlass" && "$REPO_COMPONENT" != "dkg_ir" ]]; then
  echo "Error: Invalid repo_component: '$REPO_COMPONENT'"
  echo "  Only cutlass and dkg_ir are supported"
  exit 1
fi

echo "Using GPU (SM compute compatibility): ${CUDA_SM}"
echo "Using repository component: ${REPO_COMPONENT} (cutlass for cutlass/cask, dkg_ir for DKG *_ir components)"

#----------------------------------------------------------
# Helper Functions
#----------------------------------------------------------

# Helper function to achieve a dynamic heartbeat to indicate a running process status
heartbeat () {
  WARCHING_PROCESS=$1
  WIP_INFO="${2}"
  FINISH_INFO="${3}"
  dots=""
  while kill -0 $1 2>/dev/null; do  # Check if the process is still running
    echo -n "${dots} ${WIP_INFO}"  # Print the heartbeat
    sleep 1  # Wait for 1 second
    echo -ne "\r"  # Move the cursor to the beginning of the line
    dots="${dots}."
  done
  echo "${dots} ${FINISH_INFO}"  # Print completion message once the extraction is done
}

# Helper function to check if an environment variable is set and not empty.
# If the variable is not set or has an empty value, 
# it outputs an error message indicating which environment variable is missing,
# and then exits the script with a status code of 1
verify_env () {
  KEY=$1
  if [ -z "${!KEY}" ]; then
    echo "${KEY} is not set!"
    exit 1
  fi
}

# Helper function to check if a file exists at the specified path. 
# If the file does not exist, the function outputs an error message indicating
# that the file was not found and exits the script with a status code of 1
verify_exist () {
  if [ ! -f "$1" ]; then
    echo "$1 does not exist!"
    exit 1
  fi
}

# Helper function to download a file from a specified URL, and saves it to a specified destination path.
# If the target directory where the file is to be saved does not exist, it will be created.
download_artifact () {
  ARTIFACT_DIR=$(dirname "$2")
  if [ ! -d "$ARTIFACT_DIR" ]; then
    mkdir -p "$ARTIFACT_DIR"
  fi
  curl -o "$2" -L "$1"
}

# Helper function to extract the contents of a .tar.gz archive file to a specified directory.
# If the target directory does not exist, it will be created.
extract_artifact () {
  if [ ! -d "$2" ]; then
    mkdir -p "$2"
  fi
  tar -zxf "$1" -C "$2" &
  tar_pid=$!
  heartbeat $tar_pid "extracting in progress" "completed unzipping tarball"
  wait $tar_pid
}

# Helper function to print "->" periodically
print_chars_with_delay () {
  word=$1
  for ((i=0; i<${#word}; i++)); do
    echo -n "${word:i:1}"  # Print each character without a newline
    sleep 0.05             # Wait for 0.05 seconds
  done
  echo  # Move to the next line after the loop finishes
}

# Helper function to print a message, conditionally using print_chars_with_delay
print_message () {
  # Temporarily disable verbosity
  set +x
  if [ -n "${VERBOSE+x}" ]; then
    print_chars_with_delay "$1"
    # Re-enable verbosity (only in verbose mode)
    set -x
  else
    echo "$1"
  fi
}

#----------------------------------------------------------
# General Environment Initialization
#----------------------------------------------------------

print_message "Starting to initialize global environment ..."

verify_env "URM_USER"
verify_env "URM_TOKEN"

CURRENT_SCRIPT_PATH=$(readlink -f "$0")
CURRENT_SCRIPT_DIR=$(dirname "${CURRENT_SCRIPT_PATH}")

# Setup environment variables, with a fall back to default value if not already set.
ARTIFACT_REPO=${ARTIFACT_REPO:="https://$URM_USER:$URM_TOKEN@urm.nvidia.com/artifactory/sw-fastkernels-generic-local"}
export ARTIFACT_DIR=${ARTIFACT_DIR:=$(pwd)/artifacts}
export EXTERNAL_DIR=${EXTERNAL_DIR:=$(pwd)/external}
mkdir -p "$ARTIFACT_DIR"
mkdir -p "$EXTERNAL_DIR"

# Download jq (a command-line tool for parsing, filtering amd manipulating JSON data) if not installed
if ! which jq-linux64 &> /dev/null ; then
  ARTIFACT_URL="$ARTIFACT_REPO/tools/jq/jq-linux64"
  ARTIFACT_PATH="$ARTIFACT_DIR/jq-linux64"

  if [ ! -f "$ARTIFACT_PATH" ]; then
    download_artifact "$ARTIFACT_URL" "$ARTIFACT_PATH"
    chmod +x "$ARTIFACT_PATH"
  fi

  export PATH=$ARTIFACT_DIR:$PATH
  echo "Using $(which jq-linux64)"
fi

# Download ctest if not installed
if ! which ctest &> /dev/null ; then
  ARTIFACT_URL="$ARTIFACT_REPO/cmake/3.17/x86_64/linux/release/cmake-3.17.5-Linux-x86_64.tar.gz"
  ARTIFACT_PATH="$ARTIFACT_DIR/cmake-3.17.tgz"
  EXTERNAL_PATH="$EXTERNAL_DIR/cmake-3.17"

  if [ ! -f "$ARTIFACT_PATH" ]; then
    download_artifact "$ARTIFACT_URL" "$ARTIFACT_PATH"
  fi

  if [ ! -d "$EXTERNAL_PATH" ]; then
    extract_artifact "$ARTIFACT_PATH" "$EXTERNAL_PATH"
  fi

  export PATH=${EXTERNAL_DIR}/cmake-3.17/bin:$PATH
  echo "Using $(which ctest)"
fi

#----------------------------------------------------------
# Amodel Environment Initialization
#----------------------------------------------------------

print_message "Starting to initialize amodel environment ..."

# Define common bloom paths amodel
CUTLASS_BLOOM_AMODEL_PATH="${CURRENT_SCRIPT_DIR}/../../../../bloom/testing/test_configuration/amodel/${CUDA_SM}"
DKG_IR_BLOOM_AMODEL_PATH="${CURRENT_SCRIPT_DIR}/../../../../../bloom/test_configuration/amodel/${CUDA_SM}"

# Locate default amodel environment configurations based on repo_component
if [[ "$REPO_COMPONENT" == "cutlass" ]]; then
  # For cutlass or cask tests on amodel, use the cutlass bloom path
  if [ -d "${CUTLASS_BLOOM_AMODEL_PATH}" ]; then
    # Support running directly from cutlass source
    DEFAULT_AMODEL_CONFIG_DIR=$(readlink -f "${CUTLASS_BLOOM_AMODEL_PATH}")
  else
    # Otherwise, assume we're in the install location and the config is
    # up one directory from our "bin" folder.
    DEFAULT_AMODEL_CONFIG_DIR=$(readlink -f "${CURRENT_SCRIPT_DIR}/..")
  fi
elif [[ "$REPO_COMPONENT" == "dkg_ir" ]]; then
  # For dkg_ir, use the path with one more level up -- root bloom path
  if [ -d "${DKG_IR_BLOOM_AMODEL_PATH}" ]; then
    # Support running directly from cutlass source
    DEFAULT_AMODEL_CONFIG_DIR=$(readlink -f "${DKG_IR_BLOOM_AMODEL_PATH}")
  else
    echo "Error: ${DKG_IR_BLOOM_AMODEL_PATH} does not exist!"
    exit 1
  fi
else
  echo "Error: Unsupported repo_component: '$REPO_COMPONENT'"
  echo "  Supported components are: cutlass, dkg_ir"
  exit 1
fi

# Locate amodel_env.json
export AMODEL_CONFIG_DIR=${AMODEL_CONFIG_DIR:=${DEFAULT_AMODEL_CONFIG_DIR}}
AMODEL_ENV_JSON_PATH=${AMODEL_CONFIG_DIR}/amodel_env.json
verify_exist "${AMODEL_ENV_JSON_PATH}"
echo "AMODEL_ENV_JSON_PATH (path to arch-specific amodel_env.json): ${AMODEL_ENV_JSON_PATH}"

# Initialize amodel log directory
export AMODEL_LOG_DIR=${AMODEL_LOG_DIR:="${AMODEL_CONFIG_DIR}"}
echo "AMODEL_LOG_DIR: $AMODEL_LOG_DIR"
mkdir -p "$AMODEL_LOG_DIR"

# Locate AmodelKnobs.txt from a set of default locations
if [ -f "${AMODEL_LOG_DIR}/AModelKnobs.txt" ]; then
  CUTLASS_AMODEL_DEFAULT_KNOB_FILE=${AMODEL_LOG_DIR}/AModelKnobs.txt
elif [ -f "$(pwd)/AModelKnobs.txt" ]; then
  CUTLASS_AMODEL_DEFAULT_KNOB_FILE=$(pwd)/AModelKnobs.txt
elif [ -f "${CURRENT_SCRIPT_DIR}/../AModelKnobs.txt" ]; then
  # When this script is installed to <install-dir>/test/cutlass/bin,
  # the AModelKnobs.txt is one directory up.
  CUTLASS_AMODEL_DEFAULT_KNOB_FILE=$(readlink -f "${CURRENT_SCRIPT_DIR}/../AModelKnobs.txt")
elif [ -f "${CURRENT_SCRIPT_DIR}/../../../../cmake/AModelKnobs.txt" ]; then
  # When the script is run from the source directory, the default knob
  # file is located in a separate path relative to the cutlass root
  # directory.
  CUTLASS_AMODEL_DEFAULT_KNOB_FILE=$(readlink -f "${CURRENT_SCRIPT_DIR}/../../../../cmake/AModelKnobs.txt")
fi

export AMODEL_OVERRIDE_DEFAULT_KNOB_FILE=${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE:=${CUTLASS_AMODEL_DEFAULT_KNOB_FILE}}
verify_exist "${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE}"
echo "amodel knobs found under path: ${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE}"

if [[ "$REPO_COMPONENT" == "cutlass" && "$CUDA_SM" == "sm107" ]]; then
  # Enable 288KB TMEM for Rubin for cutlass/cask tests on amodel
  if ! grep -q "GpuConfig::tensorMemorySizeInKB 288" "${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE}"; then
    echo "GpuConfig::tensorMemorySizeInKB 288" >> "${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE}"
  fi
fi
# Print the AModelKnobs.txt content
echo "AModelKnobs.txt content:"
sed 's/^/  /' "${AMODEL_OVERRIDE_DEFAULT_KNOB_FILE}"

# Populating environment from amodel_env.json
echo "Populating environment from amodel_env.json ..."

for k in $(jq-linux64 -r 'keys | .[]' "$AMODEL_ENV_JSON_PATH"); do
  key=$k
  value=$(jq-linux64 -r ."$k" "$AMODEL_ENV_JSON_PATH")
  if [[ -v $key ]]; then
    echo "  Variable '$key' is already defined in the environment"
    echo "    Ignored: $value"
    continue
  fi
  export "$key"="${!key:=$value}"
  echo "  $key=$value"
done

echo "Populating environment from amodel_env.json ... Done!"

# SCRATCH_DIR is used to store the intermediate artifacts downloaded/generated.
SCRATCH_DIR=${SCRATCH_DIR:=$(pwd)}
echo "SCRATCH_DIR=$SCRATCH_DIR"

if [ ! -d "$SCRATCH_DIR" ]; then
  # Only attempt to create scratch dir if it doesn't exist, otherwise
  # we can get an error if it exists but is read-only and we never
  # actually needed to write anything.
  echo "$SCRATCH_DIR does not exists, creating one to store the intermediate artifacts downloaded or generated"
  mkdir -p "$SCRATCH_DIR"
fi

# Initialize nv_amodel.so
CUDA_AMODEL_ARCH=${CUDA_AMODEL_ARCH:=blackwell}
export AMODEL_DIR=${AMODEL_DIR:=$SCRATCH_DIR/amodel/${CUDA_AMODEL_ARCH}/$CUDA_AMODEL_REVISION}
echo "AMODEL_DIR (base directory to nv_amodel.so): $AMODEL_DIR"

# Download amodel artifact if not staged
if [ ! -d "$AMODEL_DIR" ]; then
  echo CUDA_AMODEL_ARCH: "${CUDA_AMODEL_ARCH}"
  echo AMODEL_ARTIFACT_FILE: "${AMODEL_ARTIFACT_FILE:=amodel-${CUDA_AMODEL_ARCH}-${CUDA_AMODEL_REVISION}.tgz}"
  echo AMODEL_ARTIFACT: "${AMODEL_ARTIFACT:=${ARTIFACT_DIR}/amodel/${CUDA_AMODEL_ARCH}/${AMODEL_ARTIFACT_FILE}}"
  echo AMODEL_ARTIFACT_URL: "${AMODEL_ARTIFACT_URL:=${ARTIFACT_REPO}/amodel/${CUDA_AMODEL_ARCH}/x86_64/linux/release/${AMODEL_ARTIFACT_FILE}}"

  if [ ! -f "${AMODEL_ARTIFACT}" ]; then
    download_artifact "$AMODEL_ARTIFACT_URL" "$AMODEL_ARTIFACT"
  fi

  if [ ! -f "$AMODEL_DIR"/nv_amodel.so ]; then
    extract_artifact "$AMODEL_ARTIFACT" "$AMODEL_DIR"
  fi
fi

export CUDA_AMODEL_DLL=${CUDA_AMODEL_DLL:="$AMODEL_DIR"/nv_amodel.so}
[ -f "$CUDA_AMODEL_DLL" ] || (echo "Error: CUDA_AMODEL_DLL \"$CUDA_AMODEL_DLL\" not found!" 1>&2 && exit 1)
echo "CUDA_AMODEL_DLL: $CUDA_AMODEL_DLL"

# Initialize cuda driver
CUDA_DRIVER_BRANCH=${CUDA_DRIVER_BRANCH:=bugfix_main}

if [[ "${CUDA_DRIVER_BRANCH}" == "bugfix_main" ]]; then
  CUDA_DRIVER_TARBALL_NAME="cuda_driver-${CUDA_DRIVER_BRANCH}-${CUDA_DRIVER_REVISION}.tgz"
else
  CUDA_DRIVER_TARBALL_NAME="cuda_driver-${CUDA_DRIVER_REVISION}.tgz"
fi

export CUDA_DRIVER_DIR=${CUDA_DRIVER_DIR:=$SCRATCH_DIR/driver/$CUDA_DRIVER_BRANCH/$CUDA_DRIVER_REVISION}
echo "CUDA_DRIVER_DIR: $CUDA_DRIVER_DIR"

# Download cuda driver artifact if not staged
if [ ! -d "$CUDA_DRIVER_DIR" ]; then
  echo CUDA_DRIVER_BRANCH: "${CUDA_DRIVER_BRANCH}"
  echo CUDA_DRIVER_BUILD_TYPE: "${CUDA_DRIVER_BUILD_TYPE:=release}"
  echo CUDA_DRIVER_ARTIFACT: "${CUDA_DRIVER_ARTIFACT:=${ARTIFACT_DIR}/driver/${CUDA_DRIVER_BRANCH}/${CUDA_DRIVER_TARBALL_NAME}}"
  echo CUDA_DRIVER_ARTIFACT_URL: "${CUDA_DRIVER_ARTIFACT_URL:=${ARTIFACT_REPO}/cuda_driver/${CUDA_DRIVER_BRANCH}/x86_64/linux/${CUDA_DRIVER_BUILD_TYPE}/${CUDA_DRIVER_TARBALL_NAME}}"

  if [ ! -f "${CUDA_DRIVER_ARTIFACT}" ]; then
    download_artifact "$CUDA_DRIVER_ARTIFACT_URL" "$CUDA_DRIVER_ARTIFACT"
  fi

  if [ ! -f "$CUDA_DRIVER_DIR"/nvidia-smi ]; then
    extract_artifact "$CUDA_DRIVER_ARTIFACT" "$CUDA_DRIVER_DIR"
    chmod +x "$CUDA_DRIVER_DIR"/*.run
    (cd "$CUDA_DRIVER_DIR" && bash ./*.run -x && find . -maxdepth 1 -type d -iname 'NVIDIA*' -exec rsync -a --remove-source-files {}/ ./ \; -exec rm -r {} \;)
  fi
fi

# Create the symbolic link to the libcuda.so file
LIB_CUDA_PATH=$(find "$CUDA_DRIVER_DIR" -iname 'libcuda.so*' -print | head -n 1)
[ -f "$LIB_CUDA_PATH" ] || [ -L "$LIB_CUDA_PATH" ] || (echo "Error: libcuda.so variant could not be found in $CUDA_DRIVER_DIR" >&2 && exit 1)
[ -f "$CUDA_DRIVER_DIR"/libcuda.so.1 ] || [ -L "$CUDA_DRIVER_DIR"/libcuda.so.1 ] || ln -s "$LIB_CUDA_PATH" "$CUDA_DRIVER_DIR"/libcuda.so.1

# Create the symbolic link to the libnvidia-ptxjitcompiler.so file (WAR for https://jirasw.nvidia.com/browse/CFK-22873)
PTX_JIT_PATH=$(find "$CUDA_DRIVER_DIR" -iname 'libnvidia-ptxjitcompiler.so*' -print | head -n 1)
[ -f "$PTX_JIT_PATH" ] || [ -L "$PTX_JIT_PATH" ] || (echo "Error: libnvidia-ptxjitcompiler.so variant could not be found in $CUDA_DRIVER_DIR" >&2 && exit 1)
[ -f "$CUDA_DRIVER_DIR"/libnvidia-ptxjitcompiler.so.1 ] || [ -L "$CUDA_DRIVER_DIR"/libnvidia-ptxjitcompiler.so.1 ] || ln -s "$PTX_JIT_PATH" "$CUDA_DRIVER_DIR"/libnvidia-ptxjitcompiler.so.1

# Initialize cuda toolkit
CUDA_TOOLKIT_VARIANT=${CUDA_TOOLKIT_VARIANT:=gpgpu_internal}
CUDA_TOOLKIT_BRANCH=${CUDA_TOOLKIT_VARIANT%%_*}
CUDA_TOOLKIT_FLAVOR=${CUDA_TOOLKIT_VARIANT#*_}

# Special handle for dev toolkit (gpgpu, gpgpu-ext)
if [[ "${CUDA_TOOLKIT_BRANCH}" == "gpgpu" || "${CUDA_TOOLKIT_BRANCH}" == "gpgpu-ext" ]]; then
  CUDA_TOOLKIT_TARBALL_NAME="cuda-${CUDA_TOOLKIT_BRANCH}-${CUDA_REVISION}.tgz"
else
  CUDA_TOOLKIT_TARBALL_NAME="cuda-${CUDA_REVISION}.tgz"
fi

# Special handle for external toolkit
if [[ "${CUDA_TOOLKIT_FLAVOR}" == "external" ]]; then
  CUDA_TOOLKIT_ARTIFACTORY_ADAPTER="release"
else
  CUDA_TOOLKIT_ARTIFACTORY_ADAPTER="release-${CUDA_TOOLKIT_FLAVOR}"
fi

export CUDA_PATH=$SCRATCH_DIR/cuda/$CUDA_TOOLKIT_VARIANT/$CUDA_REVISION
echo "CUDA_PATH: $CUDA_PATH"

# Download the cuda toolkit artifact if not staged
if [ ! -d "$CUDA_PATH" ]; then
  echo CUDA_TOOLKIT_VARIANT: "${CUDA_TOOLKIT_VARIANT}"
  echo CUDA_ARTIFACT: "${CUDA_ARTIFACT:=${ARTIFACT_DIR}/cuda/${CUDA_TOOLKIT_TARBALL_NAME}.tgz}"
  echo CUDA_ARTIFACT_URL: "${CUDA_ARTIFACT_URL:=${ARTIFACT_REPO}/cuda/${CUDA_TOOLKIT_BRANCH}/x86_64/linux/generic/${CUDA_TOOLKIT_ARTIFACTORY_ADAPTER}/${CUDA_TOOLKIT_TARBALL_NAME}}"

  if [ ! -f "$CUDA_ARTIFACT" ]; then
    download_artifact "$CUDA_ARTIFACT_URL" "$CUDA_ARTIFACT"
  fi

  if [ ! -d "$CUDA_PATH"/lib64 ]; then
    extract_artifact "$CUDA_ARTIFACT" "$CUDA_PATH"
  fi
fi

# Set global binary and library loading path
export LD_LIBRARY_PATH=$CUDA_DRIVER_DIR:$CUDA_PATH/lib64:$CXX_BASE/$CXX_REVISION/lib64:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
export PATH=$CMAKE_BASE/$CMAKE_REVISION/bin:$PATH
echo "PATH: $PATH"

#----------------------------------------------------------
# Execute Tests on Amodel
#----------------------------------------------------------

print_message "Starting to run tests on amodel ..."

export CUDA_DISABLE_PTX_JIT=1
export CUDA_CACHE_DISABLE=1

echo LOAD_EXE_BEFORE_EXECUTION=1 "$@"
LOAD_EXE_BEFORE_EXECUTION=1 "$@"
