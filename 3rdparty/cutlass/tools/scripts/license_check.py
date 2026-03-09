#! /usr/bin/env python

###################################################################################################
# Copyright (C) 2024 - 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
###################################################################################################```
#
#  {$nv-release-never file} - guard to avoid releasing the release script
#
# Script to check that files have license


import argparse
import os


def check_license(directory):
    license_lines = [
        'Copyright (c)',
        'NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
        'SPDX-License-Identifier: BSD-3-Clause',
        'Redistribution and use in source and binary forms, with or without',
        'modification, are permitted provided that the following conditions are met:',
        '1. Redistributions of source code must retain the above copyright notice, this',
        'list of conditions and the following disclaimer.',
        '2. Redistributions in binary form must reproduce the above copyright notice,',
        'this list of conditions and the following disclaimer in the documentation',
        'and/or other materials provided with the distribution.',
        '3. Neither the name of the copyright holder nor the names of its',
        'contributors may be used to endorse or promote products derived from',
        'this software without specific prior written permission.',
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        'AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE',
        'IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE',
        'DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE',
        'FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL',
        'DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR',
        'SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER',
        'CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,',
        'OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE',
        'OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.',
    ]

    # .md files are not included in this list because we currently have many Markdown documents
    # for documentation that do not contain licenses.
    #
    # .txt files are not included in this list because we currently keep many hashes in such files.
    extensions = (".h", ".hpp", ".cmake", ".cu", ".cpp", ".sh", ".py")

    skip_list = [
        "CONTRIBUTORS.md",
        "PUBLICATIONS.md"
    ]

    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in files:
            if not filename.endswith(extensions) or filename in skip_list:
                continue

            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()

            # Check that each line of the license is included at least somewhere. While this is
            # not fool-proof, it seems unlikely that we'd have a file with these lines spread throughout
            # or in a different order.
            for license_line in license_lines:
                if license_line not in s:
                    raise Exception(f"Incorrect license detected in file {filepath}. Expected line {license_line} to appear.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cutlass_root", type=str,
                        help="Path to CUTLASS repository")
    args = parser.parse_args()

    check_license(args.cutlass_root)


if __name__ == "__main__":
    main()