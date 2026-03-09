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
# Script to construct a CUTLASS release from a Git repository.
#
# This script recurses through a white-listed directory structure, copying and filtering files to
# form a copy of the repository which may be released. The script applies a filter option to the
# line of code it appears and possibly a larger scope according to the syntax below.
#
# The string matching regex "\{\$nv-.+\}" should be avoided EXCEPT where source filters are
# intended.
#
#  Filter syntax in source code:
#
#    {$<filter>}          # filter has line scope
#    {$<filter> file}     # filter has file scope
#    {$<filter> begin}    # start of filter region
#    {$<filter> end}      # end of filter region
#
#  filters:
#    nv-license
#    nv-internal-release
#    nv-nda-release
#    nv-internal-experimental
#    nv-release-never
#    nv-public-only
#    nv-internal-release-only
#    nv-TODO
#    nv-cicd
#    nv-profiling
#
#  release type:
#    public
#      included: {nv-public-only, nv-TODO}
#
#    nda
#      included: {nv-public-only, nv-nda-release, nv-TODO}
#
#    public-cicd
#      included: {nv-public-only, nv-TODO, nv-cicd}
#
#    profiling
#      included: {nv-public-only, nv-profiling, nv-TODO}
#
#    internal
#      included: {nv-internal-release, nv-internal-release-only, nv-TODO}
#
#    experimental
#      included: {nv-internal-release, nv-internal-experimental, nv-TODO}
#
#
###################################################################################################

import os
import shutil
import sys
import re
import argparse

#
def str_to_bool(x):
    """ Returns false if the string is 0, false, or False. Otherwise, returns True."""
    return x is not None and x not in ('0', 'false', 'False')

#
class Options:
    """ Options for release automation. """
    #
    def __init__(self):
        self.dst_path = '..'
        self.src_path = 'cutlass/'
        self.release_type = 'public'
        self.line_end = 'unix'
        self.license_file = 'LICENSE.txt'
        self.target_name = ''
        self.overwrite = True
        self.verbose = True
        self.doc_only = False
        self.copy_dsl = False
        self.dsl_release_branch = ''
        self.dsl_release_commit = ''

        self.parser = argparse.ArgumentParser(description='CUTLASS release automation')
        self.parser.add_argument('--dst_path', dest='dst_path', metavar='<str>', type=str,
            default=self.dst_path, help='Path to directory in which release directory is created.')
        self.parser.add_argument('--src_path', dest='src_path', metavar='<str>', type=str,
            default=self.src_path, help='Path to CUTLASS project directory.')
        self.parser.add_argument('--release_type', dest='release_type', metavar='<str>', type=str,
            default=self.release_type, help='Release type: public|profiling|internal|experimental|public-cicd|cutensor-JIT|nda')
        self.parser.add_argument('--line_end', dest='line_end', metavar='<src>', type=str,
            default=self.line_end, help="Line ending character: unix*|dos|mac")
        self.parser.add_argument('--license_file', dest='license_file', metavar='<str>', type=str,
            default=self.license_file, help="Name of license file in repository")
        self.parser.add_argument('--target_name', dest='target_name', metavar='<str>', type=str,
            default=self.target_name, help="Name of release")
        self.parser.add_argument('--force', '--overwrite', dest='overwrite', metavar='<bool>', type=str,
            default=str(self.overwrite), help="If true and destination directory exists, it is completely overwritten.")
        self.parser.add_argument('--verbose', dest='verbose', metavar='<bool>', type=str,
            default=str(self.verbose), help="Prints verbose logging of filtering operations.")
        self.parser.add_argument('--doc_only', dest='doc_only', metavar='<bool>', type=str,
            default=str(self.doc_only), help="If true, only process the media/docs folder.")
        self.parser.add_argument('--copy_dsl', dest='copy_dsl', metavar='<bool>', type=str,
            default=str(self.copy_dsl), help="If true, copy dsl release codes to cutlass folder.")
        self.parser.add_argument('--dsl_release_branch', dest='dsl_release_branch', metavar='<str>', type=str,
            default=self.dsl_release_branch, help='Name of dsl release branch to copy if copy_dsl is true.')
        self.parser.add_argument('--dsl_release_commit', dest='dsl_release_commit', metavar='<str>', type=str,
            default=self.dsl_release_commit, help='Commit id of dsl release branch to copy if copy_dsl is true.')

        self.args = self.parser.parse_args()

        map_line_ending = {
            'unix': "\n",
            'dos': "\r\n",
            'mac': "\r"
        }

        self.dst_path = self.args.dst_path
        self.src_path = self.args.src_path
        self.release_type = self.args.release_type
        self.line_end = map_line_ending[self.args.line_end]
        self.license_file = self.args.license_file

        self.target_name = self.args.target_name if self.args.target_name != '' else "cutlass_%s_rc" % self.args.release_type

        self.src_extensions = [".cpp", ".cuh", ".hpp", ".txt", ".cu", ".md", ".py", ".c", ".h"]
        self.bin_extensions = ['.png', '.bin', '.exe', '.git']
        self.blacklisted_extensions = ['.pyc',]
        self.overwrite = str_to_bool(self.args.overwrite)
        self.verbose = str_to_bool(self.args.verbose)
        self.doc_only = str_to_bool(self.args.doc_only)
        self.copy_dsl = str_to_bool(self.args.copy_dsl)
        self.dsl_release_branch = self.args.dsl_release_branch
        self.dsl_release_commit = self.args.dsl_release_commit

    #
    def action(self):
        """ Indicates what kind of action to take. """
        return self.release_type

    #
    def dst_root(self):
        """ Returns the root directory for the release"""
        return os.path.join(self.dst_path, self.target_name)

    #
    def _has_extension(self, path, extensions):
        """ Returns true if file ends with any extension in a list."""
        for extn in extensions:
            if path.endswith(extn):
                return True
        return False

    #
    def is_bin_file(self, src_path):
        """ Returns true if the path corresponds to a binary file. """
        return self._has_extension(src_path, self.bin_extensions)

    #
    def is_src_file(self, src_path):
        """ Returns true if the given path ends with an extension corresponding to
            source code that appears in CUTLASS. """
        #return self._has_extension(src_path, self.src_extensions)
        return not self.is_bin_file(src_path)

    #
    def is_blacklisted_file(self, src_path):
        """ Returns true if the path corresponds to a binary file. """
        return self._has_extension(src_path, self.blacklisted_extensions)

#
class CutlassSource:
    """ """
    def __init__(self, release):
        self.release = release

        examples_for_release = [
          ('00_basic_gemm', '*'),
          ('01_cutlass_utilities', '*'),
          ('02_dump_reg_shmem', '*'),
          ('03_visualize_layout', '*'),
          ('04_tile_iterator', '*'),
          ('05_batched_gemm', '*'),
          ('06_splitK_gemm', '*'),
          ('07_volta_tensorop_gemm', '*'),
          ('08_turing_tensorop_gemm', '*'),
          ('09_turing_tensorop_conv2dfprop', '*'),
          ('10_planar_complex', '*'),
          ('11_planar_complex_array', '*'),
          ('12_gemm_bias_relu', '*'),
          ('13_two_tensor_op_fusion', '*'),
          ('14_ampere_tf32_tensorop_gemm', '*'),
          ('15_ampere_sparse_tensorop_gemm', '*'),
          ('16_ampere_tensorop_conv2dfprop', '*'),
          ('17_fprop_per_channel_bias', '*'),
          ('18_ampere_fp64_tensorop_affine2_gemm', '*'),
          ('19_tensorop_canonical', '*'),
          ('20_simt_canonical', '*'),
          ('21_quaternion_gemm', '*'),
          ('22_quaternion_conv', '*'),
          ('23_ampere_gemm_operand_reduction_fusion', '*'),
          ('24_gemm_grouped', '*'),
          ('25_ampere_fprop_mainloop_fusion', '*'),
          ('26_ampere_wgrad_mainloop_fusion', '*'),
          ('27_ampere_3xtf32_fast_accurate_tensorop_gemm', '*'),
          ('28_ampere_3xtf32_fast_accurate_tensorop_fprop', '*'),
          ('29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm', '*'),
          ('30_wgrad_split_k', '*'),
          ('31_basic_syrk', '*'),
          ('32_basic_trmm', '*'),
          ('33_ampere_3xtf32_tensorop_symm', '*'),
          ('34_transposed_conv2d', '*'),
          ('35_gemm_softmax', '*'),
          ('36_gather_scatter_fusion', '*'),
          ('37_gemm_layernorm_gemm_fusion', '*'),
          ('38_syr2k_grouped', '*'),
          ('39_gemm_permute', '*'),
          ('40_cutlass_py', '*'),
          ('41_fused_multi_head_attention', '*'),
          ('42_ampere_tensorop_group_conv', '*'),
          ('43_ell_block_sparse_gemm', '*'),
          ('44_multi_gemm_ir_and_codegen', '*'),
          ('45_dual_gemm', '*'),
          ('46_depthwise_simt_conv2dfprop', '*'),
          ('47_ampere_gemm_universal_streamk', '*'),
          ('48_hopper_warp_specialized_gemm', '*'),
          ('49_hopper_gemm_with_collective_builder', '*'),
          ('50_hopper_gemm_with_epilogue_swizzle', '*'),
          ('51_hopper_gett', '*'),
          ('52_hopper_gather_scatter_fusion', '*'),
          ('53_hopper_gemm_permute', '*'),
          ('54_hopper_fp8_warp_specialized_gemm', '*'),
          ('55_hopper_mixed_dtype_gemm', '*'),
          ('56_hopper_ptr_array_batched_gemm', '*'),
          ('57_hopper_grouped_gemm', '*'),
          ('58_ada_fp8_gemm', '*'),
          ('59_ampere_gather_scatter_conv', '*'),
          ('60_cutlass_import', '*'),
          ('61_hopper_gemm_with_topk_and_softmax', '*'),
          ('62_hopper_sparse_gemm', '*'),
          ('63_hopper_gemm_with_weight_prefetch', '*'),
          ('64_ada_fp8_gemm_grouped', '*'),
          ('65_distributed_gemm', '*'),
          ('67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling', '*'),
          ('68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling', '*'),
          ('69_hopper_mixed_dtype_grouped_gemm', '*'),
          ('70_blackwell_gemm', '*'),
          ('71_blackwell_gemm_with_collective_builder', '*'),
          ('72_blackwell_narrow_precision_gemm', '*'),
          ('73_blackwell_gemm_preferred_cluster', '*'),
          ('74_blackwell_gemm_streamk', '*'),
          ('75_blackwell_grouped_gemm', '*'),
          ('76_blackwell_conv', '*'),
          ('77_blackwell_fmha', '*'),
          ('78_blackwell_emulated_bf16x9_gemm', '*'),
          ('79_blackwell_geforce_gemm', '*'),
          ('80_blackwell_geforce_sparse_gemm', '*'),
          ('81_blackwell_gemm_blockwise', '*'),
          ('82_blackwell_distributed_gemm', '*'),
          ('83_blackwell_sparse_gemm', '*'),
          ('84_blackwell_narrow_precision_sparse_gemm', '*'),
          ('86_blackwell_mixed_dtype_gemm', '*'),
          ('87_blackwell_geforce_gemm_blockwise', '*'),
          ('88_hopper_fmha', '*'),
          ('89_sm103_fp4_ultra_gemm', '*'),
          ('90_sm103_fp4_ultra_grouped_gemm', '*'),
          ('91_fp4_gemv', '*'),
          ('92_blackwell_moe_gemm', '*'),
          ('93_blackwell_low_latency_gqa', '*'),
          ('94_ada_fp8_blockwise', '*'),
          ('111_hopper_ssd', '*'),
          ('112_blackwell_ssd', '*'),
          ('common', '*'),
          ('cute', [
            ('tutorial', '*'),
            ('CMakeLists.txt', '*'),
          ]),
          ('python', '*'),
          ('CMakeLists.txt', '*'),
          ('README.md', '*'),
        ]

        self.full_whitelist = [
            ('CMake', '*'),
            ('bloom', '*'), # Needed for pipelines, will be removed by mirror.py
            ('compiler_testlists', [
                ('FK_Compiler_perf_testlist_GH100_SM90_cutlass3x_gemm.csv', '*'),
                ('FK_Compiler_perf_testlist_GH100_SM90_cutlass3x_gemm_L0.csv', '*'),
                ('FK_Compiler_perf_testlist_GH100_SM90_cutlass3x_gemm_kernel_filter.list', '*'),
                ('FK_Compiler_perf_testlist_GB100_SM100_cutlass3x_gemm_public.csv', '*')
            ]), # Needed for pipelines performance testing, will be removed by mirror.py
            ('include', [
              ('cutlass', [
                ('arch', [
                  ('*.h', '*'),
                  ('*.hpp', '*'),
                ]),
                ('detail', '*'),
                ('contraction', '*'),
                ('epilogue', '*'),
                ('conv', '*'),
                ('gemm', '*'),
                ('layout', '*'),
                ('pipeline', '*'),
                ('platform', '*'),
                ('reduction', '*'),
                ('thread', '*'),
                ('transform', '*'),
                ('util', '*'),
                ('*.h', '*'),
                ('*.hpp', '*'),
                ('experimental', '*'),
              ]),
              ('cute', [
                ('algorithm', '*'),
                ('arch', [
                  ('*.hpp', '*'),
                  ('bringup', '*'),
                ]),
                ('atom', '*'),
                ('container', '*'),
                ('numeric', '*'),
                ('util', '*'),
                ('*.hpp', '*'),
              ]),
            ]),
            ('cmake', '*'),
            ('examples', examples_for_release),
            ('media', [
              ('docs', [
                ('cpp', '*'),
              ]),
              ('images', '*'),
            ]),
            ('python', '*'),
            ('tools', [
              ('util', '*'),
              ('library', [
                ('include', [
                  ('cutlass', [
                    ('library', [
                        ('*.h', '*'),
                        ('*.hpp', '*'),
                    ]),
                  ]),
                ]),
                ('src', [
                  ('reduction', '*'),
                  ('reference', '*'),
                  ('conv2d_operation.h', '*'),
                  ('conv3d_operation.h', '*'),
                  ('conv_operation_3x.hpp', '*'),
                  ('gemm_operation.h', '*'),
                  ('gemm_operation_3x.hpp', '*'),
                  ('sparse_gemm_operation_3x.hpp', '*'),
                  ('block_scaled_gemm_operation_3x.hpp', '*'),
                  ('blockwise_gemm_operation_3x.hpp', '*'),
                  ('grouped_gemm_operation_3x.hpp', '*'),
                  ('handle.cu', '*'),
                  ('library_internal.h', '*'),
                  ('manifest.cpp', '*'),
                  ('operation_table.cu', '*'),
                  ('rank_2k_operation.h', '*'),
                  ('rank_k_operation.h', '*'),
                  ('singleton.cu', '*'),
                  ('sparse_gemm_operation_3x.hpp', '*'),
                  ('symm_operation.h', '*'),
                  ('trmm_operation.h', '*'),
                  ('util.cu', '*'),
                ]),
                ('CMakeLists.txt', '*'),
              ]),
              ('profiler', '*'),
              ('scripts', [
                ('ci', '*'),
              ]),
              ('CMakeLists.txt', '*'),
              ('googletest.cmake', '*'),
            ]),
            ('test', [
              ('python', '*'),
              ('unit', [
                ('cluster_launch', '*'),
                ('common', '*'),
                ('conv', [
                  ('CMakeLists.txt', '*'),
                  ('cache_testbed_output.h', '*'),
                  ('device', '*'),
                  # 'device_3x' is expanded out to remove 'mods' and 'grafia' subdirectories
                  ('device_3x', [
                    ('CMakeLists.txt', '*'),
                    ('*.hpp', '*'),
                    ('dgrad', '*'),
                    ('wgrad', '*'),
                    ('fprop', [
                      ('CMakeLists.txt', '*'),
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                    ]),
                  ])
                ]),
                ('contraction', '*'),
                ('core', '*'),
                ('cute', [
                  ('ampere', '*'),
                  ('core', '*'),
                  ('hopper', '*'),
                  ('layout', '*'),
                  ('msvc_compilation', '*'),
                  ('turing', '*'),
                  ('volta', '*'),
                  ('CMakeLists.txt', '*'),
                  ('*.hpp', '*'),
                ]),
                ('data', [
                  ('hashes', [
                    ('*_simt.txt', '*'),
                    ('*_sm70.txt', '*'),
                    ('*_sm75.txt', '*'),
                    ('*_sm80.txt', '*'),
                    ('*_sm90.txt', '*'),
                    ('cached_results_cutlass_test_unit_conv_device_tensorop_s32.txt', '*'),
                    ('cached_results_cutlass_test_unit_conv_device_tensorop_s32_interleaved.txt', '*'),
                  ]),
                ]),
                ('epilogue', '*'),
                ('gemm', [
                  # 'device' directory is expanded out to remove 'mods' and 'grafia' subdirectories
                  ('device', [
                    ('*.cu', '*'),
                    ('*.h', '*'),
                    ('*.hpp', '*'),
                    ('*.py', '*'),
                    ('CMakeLists.txt', '*'),
                    ('sm100_blockscaled_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ('sm100_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                      ("narrow_precision", [
                        ('*.cu', '*'),
                        ('*.h', '*'),
                        ('*.hpp', '*'),
                        ('*.py', '*'),
                        ('CMakeLists.txt', '*'),
                      ]),
                    ]),
                    ('sm100_sparse_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                      ("narrow_precision", [
                        ('*.cu', '*'),
                        ('*.h', '*'),
                        ('*.hpp', '*'),
                        ('*.py', '*'),
                        ('CMakeLists.txt', '*'),
                      ]),
                    ]),
                    ('sm100_blockscaled_sparse_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ('sm120_blockscaled_sparse_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ('sm120_sparse_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ('sm120_blockscaled_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ('sm120_tensorop_gemm', [
                      ('*.cu', '*'),
                      ('*.h', '*'),
                      ('*.hpp', '*'),
                      ('*.py', '*'),
                      ('CMakeLists.txt', '*'),
                    ]),
                    ]
                  ),
                  ('kernel', '*'),
                  ('thread', '*'),
                  ('threadblock', '*'),
                  ('warp', '*'),
                  ('CMakeLists.txt', '*'),
                ]),
                ('layout', '*'),
                ('nvrtc', '*'),
                ('pipeline', '*'),
                ('substrate', '*'),
                ('reduction', '*'),
                ('transform', '*'),
                ('util', '*'),
                ('*.txt', '*'),
                ('*.cpp', '*'),
                ('*.h', '*'),
                ('*.cu', '*'),
                ('CMakeLists.txt', '*')
              ]),
              ('self_contained_includes', [
                ('CMakeLists.txt', '*')
              ]),
              ('CMakeLists.txt', '*'),
            ]),
            ('CMakeLists.txt', '*'),
            ('CUTLASS.md', '*'),
            ('README.md', '*'),
            ('CHANGELOG.md', '*'),
            ('CONTRIBUTORS.md', '*'),
            ('CITATION.cff', '*'),
            ('PUBLICATIONS.md', '*'),
            ('feature_detect.cmake', '*'),
            ('bin2hex.cmake', '*'),
            ('CUDA.cmake', '*'),
            ('cuBLAS.cmake', '*'),
            ('cuDNN.cmake', '*'),
            ('customConfigs.cmake', '*'),
            ('Jenkinsfile', '*'),
            ('LICENSE.txt', '*'),
            ('Doxyfile', '*'),
            ('feature_detect.cmake', '*'),
            ('revisions.cmake', '*'),
            ('pyproject.toml', '*'),
            ('setup.cfg', '*'),
            ('.gitignore', '*'),
            ('.github', '*')
        ]

        # Set the whitelist based on doc_only flag
        if release.options.doc_only:
            self.whitelist = [
                ('media', [
                    ('docs', '*'),
                    ('images', '*'),
                ]),
                ('README.md', '*'),
                ('CHANGELOG.md', '*'),
                ('index.rst', '*'),
            ]
        else:
            self.whitelist = self.full_whitelist

    #
    def _visit_item(self, item_path, grandchildren, options, func):

        head, tail = os.path.split(item_path)

        if tail.startswith("*"):
          for item in os.listdir(os.path.join(options.src_path, head)):
            if item != '.' and item != '..' and item.endswith(tail[1:]):
              self._visit_item(os.path.join(head, item), grandchildren, options, func)
        else:
          item_abs_path = os.path.join(options.src_path, item_path)
          if os.path.isfile(item_abs_path):
              func(item_path, options)
          elif os.path.isdir(item_abs_path):
              self._visit_dir(item_path, grandchildren, options, func)

    #
    def _visit_dir(self, dir_path, children, options, func):
        """ Visit a directory and its children """

        # Look for a release control file

        include_directory = True

        this_dir_path = os.path.join(options.src_path, dir_path)
        dir_control = os.path.join(this_dir_path, ".cutlass-release-control")
        if os.path.isfile(dir_control):
          lines = []
          with open(dir_control, 'r') as control_file:
            lines = [line.rstrip() for line in control_file]
            any_filters_found = False
            filters_found = []

            for line in lines:
              match = self.release.filter_tag.search(line)
              if match is not None:
                any_filters_found = True
                filter_text_items = match.group(1).split(' ')
                filters_found.append(filter_text_items[0])

            if any_filters_found:
              include_directory = False
              for filter_text in filters_found:
                if filter_text in self.release.get_active_filters():
                  include_directory = True

        if not include_directory:
          return

        # create directory at destination if does not exist
        dst_dir_path = os.path.join(options.dst_root(), dir_path)
        if not os.path.isdir(dst_dir_path):
            os.mkdir(dst_dir_path)

        if isinstance(children, str) and children.startswith('*'):
            # exhaustive enumeration
            for item in os.listdir(os.path.join(options.src_path, dir_path)):
                if item != '.' and item != '..' and item.endswith(children[1:]):
                    self._visit_item(os.path.join(dir_path, item), children, options, func)
        elif children is None:
            # do not recurse
            pass
        else:
            # enumeration based on whitelist
            for item, grandchildren in children:
                self._visit_item(os.path.join(dir_path, item), grandchildren, options, func)

    #
    def visit(self, options, func):
        """ """
        return self._visit_dir('.', self.whitelist, options, func)

#
class CutlassFileExcluded(Exception):
    def __init__(self, filter_text):
        self.filter_text = filter_text

#
class CutlassRelease:
    """ Class to construct a release of CUTLASS. """

    #
    def __init__(self, options=Options()):
        self.options = options
        self.source = CutlassSource(self)

        self.filter_tag = re.compile(r"\{\$nv-(.+)\}")
        # `release_filters` means those release guard will be release
        # e.f. `internal` means `nv-internal-release` guarded code will ba ADDED (not filtered)
        self.release_filters = {
            'public': [ 'license', 'public-only', 'TODO' ],
            'nda': [ 'license', 'public-only', 'nda-release', 'TODO' ],
            'public-cicd': [ 'license', 'public-only', 'cicd', 'TODO' ],
            'cutensor-JIT': ['license', 'public-only', 'TO'+'DO', 'cutensor-JIT'],
            'profiling': [ 'license', 'public-only', 'profiling', 'TODO' ],
            'internal': [ 'license', 'internal-release', 'internal-release-only', 'profiling', 'TODO'],
            'experimental': [ 'license', 'internal-release', 'internal-experimental', 'profiling', 'TODO']
        }
        self.filter_context_stack = []
        self.previous_filter_enable = True

        self.filter_actions = {
            'license': self._generate_license,
            'profiling': None,
            'cicd': None,
            'internal-release': None,
            'nda-release': None,
            'internal-experimental': None,
            'release-never': None,
            'public-only': None,
            'internal-release-only': None,
            'TODO': None,
            'public': None,
            'cutensor-JIT': None,
        }

        self.print_guard = False

    #
    def _generate_line(self, dst_file, src_path, src_line, match = None):
        """ Writes a line of text, modifying only line endings """
        dst_line  = src_line.rstrip("\r\n") + self.options.line_end
        dst_file.write(dst_line)

    #
    def _generate_license(self, dst_file, src_path, src_line, match = None):
        """ Inserts text of license into source file """
        license_path = os.path.join(self.options.src_path, self.options.license_file)
        with open(license_path, 'r') as license_file:
            for license_line in license_file.readlines():
                self._generate_line(dst_file, src_path, license_line)

    #
    def get_active_filters(self):
        return self.release_filters[self.options.release_type]

    #
    def _evaluate_filter(self, line_scope_filter = ''):
        """ The line-scope filter and all filters on the stack must be approved for
            the given release type. """
        filter_stack = self.filter_context_stack
        if line_scope_filter:
            filter_stack = filter_stack + [line_scope_filter]
        for tag in filter_stack:
            if tag not in self.get_active_filters():
                return False
        return True

    #
    def _swallow_comment(self, src_line, match):
        """ If a match is found on a line, assume it follows a one-line comment token.
            Returns the substring that ends at the comment token so the filters do not
            appear in the release even when the source does.

            Note, this does NOT attempt to consume the block comment open token /* which
            is assumed to be unrelated to the source filter. """
        if match is not None:
            start, end = match.span()

            # first consume whitespace
            while start > 0 and src_line[start - 1] == ' ':
                start = start - 1

            # then consume consecutive identical characters (e.g. //, ##, )
            block_comment = False
            if start > 0:
                ch = src_line[start - 1]
                if ch == '*':                   # special handling of C-style /* block comments */
                    block_comment = True
                while not block_comment and start > 0 and src_line[start - 1] == ch:
                    start = start - 1

            # return line
            return src_line[:start] + src_line[end:] if block_comment else src_line[:start]

        # Pass through if no match
        return src_line

    #
    def _transform_line(self, dst_file, src_path, src_line):
        """ For a given line, copies file and filters those containing special comments. """

        self.print_guard = False

        filter_text = ''
        filter_operation = ''
        match = self.filter_tag.search(src_line) if self.options.is_src_file(src_path) else None

        if match is not None:
            filter_text_items = match.group(1).split(' ')
            filter_text = filter_text_items[0]
            filter_operation = filter_text_items[1] if len(filter_text_items) > 1 else ''
            if filter_text not in self.filter_actions.keys():
                raise Exception("Unknown source filter: %s in '%s':\n%s" % (filter_text, src_path, src_line))

        filter_enable = self._evaluate_filter(filter_text)

        if filter_enable:
            # perform action
            func = self.filter_actions[filter_text] if filter_text else None
            if func is not None:
                func(dst_file, src_path, src_line, match)

            # if the previous line was disabled, and this line contains only whitespace,
            # filter it as well
            if (self.previous_filter_enable and not filter_text) or not src_line.isspace():
              self._generate_line(dst_file, src_path, self._swallow_comment(src_line, match))

        # update filter context stack
        self.previous_filter_enable = filter_enable
        if filter_operation == 'begin':
            self.filter_context_stack.append(filter_text)
        elif filter_operation == 'end':
            try:
              self.filter_context_stack.pop()
            except:
              print("Failed to pop filtering context.")
              print("From %s[%s]" % (src_path, src_line))
              exit(1)
        elif filter_operation == 'file':
            if not filter_enable:
                raise CutlassFileExcluded(filter_text)

    #
    def transform_file(self, dst_path, src_path):
        """ Given a source and destination path, filters source file. """

        dst_abs_path = os.path.join(self.options.dst_root(), dst_path)
        src_abs_path = os.path.join(self.options.src_path, src_path)

        if self.options.verbose:
            print("transform_file(): %s" % src_path)

        # reset context
        self.filter_context_stack = []
        self.previous_filter_enable = True

        if not self.options.is_blacklisted_file(src_path):
          if self.options.is_src_file(src_path):
              # filter source files
              try:
                  with open(dst_abs_path, 'w') as dst_file, open(src_abs_path, 'r') as src_file:
                      for src_line in src_file.readlines():
                          self._transform_line(dst_file, src_path, str(src_line))
              except CutlassFileExcluded as exception:
                  # delete the destination file
                  if self.options.verbose:
                      print("  %s excluded" % src_path)
                  os.remove(dst_abs_path)
              except:
                print("Unexpected error with file '%s':" % src_path, sys.exc_info()[0])
                raise
          else:
              shutil.copyfile(src_abs_path, dst_abs_path)

        # make sure begin and end tokens match
        if len(self.filter_context_stack):
            raise Exception("Error - source file %s has a non-empty filter stack: [%s]" % (src_path, ", ".join(self.filter_context_stack)))

        # set execute privileges if necessary
        if os.path.isfile(src_abs_path) and os.path.isfile(dst_abs_path):
          src_status = os.stat(src_abs_path)
          os.chmod(dst_abs_path, src_status.st_mode)

    #
    def copy_dsl_file(self):
        """ Copy DSL files from certain commit of certain DSL release branch """

        # clone and checkout to certain commit of certain DSL release branch
        os.system('git clone ssh://git@gitlab-master.nvidia.com:12051/dlarch-fastkernels/dynamic-kernel-generator.git tmp/dkg')
        os.chdir("tmp/dkg")
        os.system('git checkout %s' % self.options.dsl_release_branch)
        os.system('git reset --hard %s' % self.options.dsl_release_commit)

        # run release.py under dkg
        os.system('python3 scripts/release.py --overwrite true')

        # copy dsl files from dkg_public_rc to cutlass_public_rc
        src_dir_path = "../dkg_public_rc"
        dst_dir_path = os.path.join("../../", self.options.dst_root())

        os.mkdir(os.path.join(dst_dir_path, "python/CuTeDSL"))
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "DkgDSL/base_dsl"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "DkgDSL/cutlass_dsl"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/cutlass"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/requirements.txt"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        # rename the file to EULA.txt to avoid confusion with the top-level LICENSE.txt
        # add EULA.txt in two places: top-level + python/CuTeDSL
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/LICENSE"), os.path.join(dst_dir_path, "python/CuTeDSL/EULA.txt")))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/LICENSE"), os.path.join(dst_dir_path, "EULA.txt")))

        os.mkdir(os.path.join(dst_dir_path, "examples/python/deprecated"))
        os.system('mv %s %s' % (os.path.join(dst_dir_path, "examples/python/*"), os.path.join(dst_dir_path, "examples/python/deprecated")))
        os.mkdir(os.path.join(dst_dir_path, "examples/python/CuTeDSL"))
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/examples/*"), os.path.join(dst_dir_path, "examples/python/CuTeDSL")))

        # for the documentation, the reference is the copy of dkg we just cloned
        # erase whatever already exists before copying
        os.system('rm -rf %s' % os.path.join(dst_dir_path, "media/docs/pythonDSL"))
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "cutlass/media/docs/pythonDSL"), os.path.join(dst_dir_path, "media/docs")))
        # erase license.rst to avoid confusion
        os.system('rm %s' % os.path.join(dst_dir_path, "media/docs/pythonDSL/license.rst"))

        # remove files which need to be excluded.
        os.system('rm %s' % os.path.join(dst_dir_path, "python/CuTeDSL/base_dsl/py.typed"))
        os.system('rm %s' % os.path.join(dst_dir_path, "python/CuTeDSL/cutlass/cute/py.typed"))
        os.system('rm %s' % os.path.join(dst_dir_path, "examples/python/CuTeDSL/notebooks/CMakeLists.txt"))
        os.system('rm %s' % os.path.join(dst_dir_path, "examples/python/CuTeDSL/notebooks/.gitignore"))
        os.system('rm -rf %s' % os.path.join(dst_dir_path, "python/CuTeDSL/base_dsl/runtime/csrc"))
        os.system('rm -rf %s' % os.path.join(dst_dir_path, "python/CuTeDSL/cutlass/dialects"))
        os.system('rm -rf %s' % os.path.join(dst_dir_path, "examples/python/CuTeDSL/internal"))
        os.system('rm -rf %s' % os.path.join(dst_dir_path, "examples/python/CuTeDSL/notebooks/startup"))

        # This section is to support pip editable install of OSS CuteDSL.
        # Align CuteDSL code structure with package structure, i.e. placing base_dsl & cutlass_dsl under CuTeDSL/cutlass.
        os.system('mv %s %s' % (os.path.join(dst_dir_path, "python/CuTeDSL/base_dsl"), os.path.join(dst_dir_path, "python/CuTeDSL/cutlass")))
        os.system('mv %s %s' % (os.path.join(dst_dir_path, "python/CuTeDSL/cutlass_dsl"), os.path.join(dst_dir_path, "python/CuTeDSL/cutlass")))
        # Add modified pyproject.toml for Develop mode and add prep_editable_install.py.
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/scripts/oss/pyproject.toml"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/python/scripts/oss/prep_editable_install.py"), os.path.join(dst_dir_path, "python/CuTeDSL")))
        # Add top-level dkg scripts (for run_pytest, test_sharding, etc.)
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "scripts"), os.path.join(dst_dir_path, "bloom/scripts")))
        # Add top level files under test/python
        os.mkdir(os.path.join(dst_dir_path, "bloom/cutlass_ir_compiler_test_python"))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/test/python/conftest.py"), os.path.join(dst_dir_path, "bloom/cutlass_ir_compiler_test_python/")))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/test/python/lit.local.cfg"), os.path.join(dst_dir_path, "bloom/cutlass_ir_compiler_test_python/")))
        os.system('cp %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/test/python/requirements.txt"), os.path.join(dst_dir_path, "bloom/cutlass_ir_compiler_test_python/")))
        # Copy test examples for OSS CI coverage
        os.system('cp -r %s %s' % (os.path.join(src_dir_path, "cutlass_ir/compiler/test/python/examples"), os.path.join(dst_dir_path, "bloom/cutlass_ir_compiler_test_python/examples")))

        # delete cloned dkg folder
        os.chdir("../../")
        os.system('rm -rf tmp/')

    #
    def release(self):
        """ Constructs a release of CUTLASS in the target directory. """

        # make the destination directory
        if os.path.isdir(self.options.dst_root()):
            if self.options.overwrite:
                shutil.rmtree(self.options.dst_root())
            else:
                raise Exception("Destination directory already exists. Re-run with --overwrite=true.")
        os.mkdir(self.options.dst_root())

        # visit sources and transforms
        self.source.visit(
            self.options,
            lambda rel_path, options:
                self.transform_file(rel_path, rel_path)
        )

        # copy dsl codes if needed
        if self.options.copy_dsl:
            self.copy_dsl_file()

#
def Main():
    """ """
    release = CutlassRelease()
    release.release()

    return 0

if __name__ == "__main__":
    sys.exit(Main())
