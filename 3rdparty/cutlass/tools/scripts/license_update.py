#! /usr/bin/env python

###################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###################################################################################################
#
#  {$nv-release-never file} - guard to avoid releasing the release script
#
# Script to update CUTLASS license.

import os
import fnmatch
import sys

#
old_license_scripts = """# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

new_license_scripts = """# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

#
old_license_source = """ * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

new_license_source = """ * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

#
old_license_md = """Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

new_license_md = """Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved."""

# Note: Before running the script, please update this path to an absolute path to the current cutlass root dir
cutlass_root_abs_dir_path = # Leaving comment so that the script errors out before updating the path

#
def updateLicense(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            if (filename == "license_update.py"):
                continue
            print("Updating: " + path + "/" + filename)
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

#
def Main():
    """ """
    print("Updating: " + os.path.abspath(cutlass_root_abs_dir_path))
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.h")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.hpp")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.cmake")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.cu")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.cuh")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.cpp")
    updateLicense(cutlass_root_abs_dir_path, old_license_source, new_license_source, "*.inl")

    updateLicense(cutlass_root_abs_dir_path, old_license_scripts, new_license_scripts, "*.sh")
    updateLicense(cutlass_root_abs_dir_path, old_license_scripts, new_license_scripts, "*.py")
    updateLicense(cutlass_root_abs_dir_path, old_license_scripts, new_license_scripts, "*.txt")

    updateLicense(cutlass_root_abs_dir_path, old_license_md, new_license_md, "*.md")

    # Few files have licenses as quoted text described in files that generate new files while build.
    # Those need to be done manually.

    return 0

if __name__ == "__main__":
    sys.exit(Main())