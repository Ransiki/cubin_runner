#!/bin/bash

# {$nv-internal-release file}

NUMARG=2
if [ $# -ne $NUMARG ]; then
    echo ""
    echo "  This script ports commits from GitHub to an (internal) branch of CUTLASS that mirrors the public repository."
    echo "  For example, if one wanted to update the internal branch public/3.0 with all commits on GitHub that took"
    echo "  place since the initial (untagged) release fo CUTLASS 3.0, the following command could be used:"
    echo "     ./update_public_branch.sh public/3.0 277bd6e5379e0c1e1eb64db1a654b30e1efddc8e"
    echo "  Here, 277bd6e5379e0c1e1eb64db1a654b30e1efddc8e is the commit hash on GitHub of the initial release of CUTLASS 3.0."
    echo ""
    echo "  Usage: $0 <public-branch-name> <first-github-commit-hash>"
    echo "    public-branch-name: name of the (internal) branch containing a public release to update (e.g., public/3.0)"
    echo "    first-github-commit-hash: commit hash on GitHub of the first commit to include in applying updates from GitHub"
    exit 1
fi

branch=$1
commit=$2

git_root=$(git rev-parse --show-toplevel)

# Clone public CUTLASS and get the diff between the `commit` and HEAD
tmpdir=$(mktemp -d)
git clone https://github.com/nvidia/cutlass $tmpdir/cutlass
cd $tmpdir/cutlass
head_commit=$(git rev-parse --verify HEAD)
tmpfile=$tmpdir/patch.diff
patch=$(git diff $commit..$head_commit > $tmpfile)

# Check out the public branch and apply the diff
cd $git_root
git checkout $branch
git apply $tmpfile