#!/bin/bash
#
#  {$nv-release-never file}
#
# This script is intended to search for patterns that should not appear in internally-merged code.
# This script is intended to be run from within CI pipelines; it is not expected to function outside
# of this.

if [ $# -ne 1 ]; then
  echo "Usage $0 <target-branch>"
  echo "  target-branch: branch into which the current branch will be merged"
  exit 1
fi

target_branch=$1

# WAR for inability to take diffs on CI machines after downloading source
git config --global --add safe.directory $(pwd)
# Similar WAR, for mono repo structure
git config --global --add safe.directory "$(dirname "$PWD")"

skip_file_diff() {
  # This is a workaround for skipping certain files in the git diff (i.e.,
  # so that violations aren't found in the checking code, which can be difficult to avoid).
  # This is a workaround for the following two git pathspec filters not working
  # in CI for some reason:
  #  git diff target/$target_branch -- ':!:tools/scripts/checkin-violations.sh'
  #  git diff target/$target_branch -- . ':(exclude)tools/scripts/checkin-violations.sh'
  #
  # git diff files produce a series of lines of the form:
  #     diff --git a/x b/x
  #     --- a/x
  #     +++ b/x
  #     +new text
  #     -old text
  # We want to remove any of the text following `diff --git a/x b/x` where 'x' is checkin-violations.sh,
  # up until the next `diff --git a/x b/x` line (which we want included in the diff).
  #
  # This awk method uses variable `display` to control whether lines are to be printed. When a `diff --git` line
  # is encountered, `display` is set to 0 if that line also contains checkin-violations.sh, and is otherwise set to 1.
  awk '{
    if (/^diff --git/) {
      if (/checkin-violations\.sh/) {
        display=0
      }
      else if (/guardwords\.sh/) {
        display=0
      }
      else if (/guardwords_nda\.sh/) {
        display=0
      }
      else if (/guardwords_internal\.sh/) {
        display=0
      }
      else if (/release\.py/) {
        display=0
      }
      else {
        display=1
      }
    };
    if (display==1) {
      print
    }
  }'
}

extract_additions() {
  # Outputs all added lines in the git diff (those beginning with '+'), but
  # with the file containing the addition preceding the line. This is done
  # to make it clear which files in the diff contains violations.
  #
  # git diff files produce a series of lines of the form:
  #     diff --git a/x.cu b/x.cu
  #     --- a/x.cu
  #     +++ b/x.cu
  #     +new text
  #     -old text
  #
  # This script would output:
  #     x.cu: new text

  # Explanation of the awk command below:
  #  /^diff --git/ { curfile = substr($3,3) }
  #    On lines starting with `diff --git`, extract the third space-delimited value (i.e., `a/x.cu`) and
  #    save all but the first 2 characters to the variable `curfile` (i.e., stripping out the 'a/').
  #  /^\+/ { print curfile ":",substr($0,2); }
  #    On lines starting with a + (additions in the MR), but not those starting with ++ (e.g., `+++ b/x.cu` lines),
  #    print the value of `curfile` and all but the first character of the current line (i.e., stripping the starting '+').
  awk '
     /^diff --git/ { curfile = substr($3,3) }
     /^\+[^\+]/ { print curfile ":",substr($0,2); }
  '
}

# Get the diff of the branch to merge and the target branch. Only consider added lines.
# We take the actual diff with target/$target_branch because, in our CI pipelines,
# the target branch is titled target/branch_name (e.g., target/dev). Exclude the current
# file from this check.
diff_file=$(mktemp)
git diff target/$target_branch -- $(pwd) | skip_file_diff | extract_additions > $diff_file

export GREP="grep --ignore-case"

# Fail if any TODO or FIXME has been added with out a jirasw.nvidia.com link in the same line
res=$(${GREP} "todo\|fixme" $diff_file | ${GREP} -v jirasw.nvidia.com)
if [ "${res}" != "" ]; then
  echo "A line was added containing TODO or FIXME without an associated jirasw.nvidia.com link:"
  printf "${res}\n"
fi

# Fail if any license is added with an end year that isn't the current year, or that does not
# contain only the current year.
cur_year=$(date +%Y)
end_on_cur_year_pattern=" - ${cur_year}"
single_year_str="Copyright \(c\) ${cur_year}[,]? NVIDIA CORPORATION"
res=$(${GREP} "Copyright (c) " $diff_file | ${GREP} -v "${end_on_cur_year_pattern}" | ${GREP} -E -v "${single_year_str}")
if [ "${res}" != "" ]; then
  echo "A license was added that does not terminate in the current year:"
  printf "${res}\n"
fi
