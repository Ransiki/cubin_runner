#! /usr/bin/env bash
set -eo pipefail

if [[ -z $GITLAB_USER || -z $GITLAB_ACCESS_TOKEN ]]; then
    echo "GITLAB_USER and GITLAB_ACCESS_TOKEN must be set as part of environment variables ..."
    exit 1
fi

DISTRIBUTION=$(( lsb_release -ds || cat /etc/*release || uname -om ) 2>/dev/null | head -n1)
WORKSPACE=''
BUILD_DLSIM='false'

echo "Linux distribution is: ${DISTRIBUTION}"

print_usage() {
    printf "Usage: bash install_dlsim.sh -w /tmp/workspace -b"
}

while getopts 'w:b' flag; do
    case "${flag}" in
        w) WORKSPACE="${OPTARG}" 
           DLSIM_DIR="$WORKSPACE/dlsim" ;;
        b) BUILD_DLSIM='true' ;;
        *) print_usage
           exit 1 ;;
    esac
done

python3 -m pip install virtualenv

if [ "$EUID" -ne 0 ]; then
    echo "Running as $(whoami) ..."
    SUDO=sudo
else
    echo "Running as root ..."
    set SUDO
fi

# Install necessary packages
if [[ "${DISTRIBUTION,,}" =~ "ubuntu"|"debian" ]]; then
    echo "Installing dependencies ..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y psmisc python3-dev
elif [[ "${DISTRIBUTION,,}" =~ "centos"|"rhel" ]]; then
    echo "Installing dependencies ..."
    $SUDO yum update -y
    $SUDO yum install -y which sudo
    $SUDO yum install -y psmisc python3-devel
else
    echo "$DISTRIBUTION is not tested to support DLSim at the moment, please add support if needed ..."
    exit 1
fi

# Clone dlsim project when needed
if [ ! -d "$DLSIM_DIR" ]; then
    git clone https://$GITLAB_USER:$GITLAB_ACCESS_TOKEN@gitlab-master.nvidia.com/arch/dlsim.git "$WORKSPACE/dlsim"
else
    echo "Skip cloning dlsim project, dlsim project already exisits at $DLSIM_DIR ..."
fi

# Build dlsim project when needed
if [ "$BUILD_DLSIM" = 'true' ]; then
    echo "Starting to build dlsim ..."
    cd $WORKSPACE
    python3 -m venv venv
    source $WORKSPACE/venv/bin/activate
    cd dlsim
    make clean-develop
else
    echo "Skip building dlsim ..."
fi
