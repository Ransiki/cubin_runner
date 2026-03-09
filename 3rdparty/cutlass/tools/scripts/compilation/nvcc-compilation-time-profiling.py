#
# {$nv-internal-release file}
#
# Setup:
#
# # Dependencies
# $ pip install pandas
#
# # Compiler
# $ export CUDACXX=
#
# Usage:
#
#  # Checkout, profile, and analyze
#  $ python3 nvcc-compilation-time-profiling.py
#
#  # Just checkout CUTLASS
#  $ python3 nvcc-compilation-time-profiling.py --phases=setup
#
#  # Just profile
#  $ python3 nvcc-compilation-time-profiling.py --phases=profile
#
#  # Just analyze
#  $ python3 nvcc-compilation-time-profiling.py --phases=analyze
#

import csv
import sys
import os
import subprocess
import argparse
import pandas
from datetime import date
import itertools
from collections import defaultdict
###############################################################################################################

#
class Options:
  def __init__(self):

    #
    # Defaults
    #

    self.node = 'local'
    self.nvcc_path = os.environ['CUDACXX'] if 'CUDACXX' in os.environ.keys() else ''
    self.nvcc_version = ''

    self.src_dir = 'src'
    self.tmp_dir = 'tmp'
    self.results_dir = 'results'

    self.concatenate_with = ''
    self.final_result = 'compilation_time.csv'

    self.cuda_path = 'https://urm.nvidia.com/artifactory/list/sw-fastkernels-generic-local/cuda/gpgpu/x86_64/linux/generic/release-internal/cuda-gpgpu-latest.tgz'

    self.cutlass_git = 'ssh://git@gitlab-master.nvidia.com:12051/dlarch-fastkernels/kernel_store.git'

    self.cutlass_path = '../../../'
    self.cutlass_branch = 'main'
    self.phases = 'setup,profile,analyze'

    self.architectures = '90a,100a'
    self.nvcc_path_list = list()

    #
    # Arguments parsing
    #
    self.parser = argparse.ArgumentParser(description='CUTLASS Compilation Time Profiling')

    self.parser.add_argument('--architectures', dest='architectures', metavar='<str>', type=str,
      default=self.architectures, help="List of architectures to profile")

    self.parser.add_argument('--node', dest='node', metavar='<str>', type=str,
      default=self.node, help='Node name')

    self.parser.add_argument('--nvcc-path', dest='nvcc_path', metavar='<str>', type=str,
      default=self.nvcc_path, help='Path to NVCC binary')

    self.parser.add_argument('--nvcc-path-list', dest='nvcc_path_list', metavar='<str>', type=str,
      default=self.nvcc_path_list, help='List of paths to NVCC compilers that need to be profiled')

    self.parser.add_argument('--nvcc-version', dest='nvcc_version', metavar='<str>', type=str,
      default=self.nvcc_version, help='Path to NVCC binary')

    self.parser.add_argument('--src-dir', dest='src_dir', metavar='<str>', type=str,
      default=self.src_dir, help='path to temporary directory to emit sources')

    self.parser.add_argument('--tmp-dir', dest='tmp_dir', metavar='<str>', type=str,
      default=self.tmp_dir, help='path to temporary directory to compile object files')

    self.parser.add_argument('--results-dir', dest='results_dir', metavar='<str>', type=str,
      default=self.results_dir, help='path to temporary directory to store timing results')

    self.parser.add_argument('--final-result', dest='final_result', metavar='<str>', type=str,
      default=self.final_result, help='Path to .CSV directory containing concatenated results')

    self.parser.add_argument('--concatenate-with', dest='concatenate_with', metavar='<str>', type=str,
      default=self.concatenate_with, help='Comma separated list of paths to concatenate with the final result')

    self.parser.add_argument('--cutlass-git', dest='cutlass_git', metavar='<str>', type=str,
      default=self.cutlass_git, help='Path to CUTLASS repository.')

    self.parser.add_argument('--cutlass-path', dest='cutlass_path', metavar='<str>', type=str,
      default=self.cutlass_path, help='Path to CUTLASS repository.')

    self.parser.add_argument('--cutlass-branch', dest='cutlass_branch', metavar='<str>', type=str,
      default=self.cutlass_branch, help='CUTLASS version string')

    self.parser.add_argument('--phases', dest='phases', metavar='<str>', type=str,
      default=self.phases, help="Phases to execute: setup,profile,analyze")

    self.args = self.parser.parse_args()

    #
    # Prost process
    #
    self.architectures = self.args.architectures.split(',')
    self.node = self.args.node
    self.nvcc_path = self.args.nvcc_path
    self.nvcc_version = self.args.nvcc_version
    self.src_dir = self.args.src_dir
    self.tmp_dir = self.args.tmp_dir
    self.results_dir = self.args.results_dir
    self.final_result = self.args.final_result
    self.concatenate_with = self.args.concatenate_with.split(',')
    self.cutlass_git = self.args.cutlass_git
    self.cutlass_path = self.args.cutlass_path
    self.cutlass_branch = self.args.cutlass_branch

    if(self.args.nvcc_path_list) :
      self.nvcc_path_list = self.args.nvcc_path_list.split(',')
    else:
      self.nvcc_path_list.append(self.nvcc_path)

    self.phases = self.args.phases.split(',')
    self.cuda_bin_path = os.path.split(self.args.nvcc_path)[0]
    self.ptxExtDesc_txt = os.path.join(self.cuda_bin_path, 'ptxExtDesc.txt')

###############################################################################################################

#
class KernelInstance:
  def __init__(self, source):
    self.source = source

###############################################################################################################

KernelCounts = [1, 2, 4, 8]

Kernels = {
  '80': [
  ],
  '90a': [
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8 :
  public cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
      '''),
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8 :
  public cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ntn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    '''),

    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8 :
  public cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_tnn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
      '''),
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8
using cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8 :
  public cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_ttn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    '''),
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8 :
  public cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_nnn_align8"));


}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
      '''),
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8 :
  public cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ntn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    '''),

    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8 :
  public cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_tnn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////


      '''),
    KernelInstance('''
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_epilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8
using cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_mainloop,
    cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8 :
  public cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x128x64_1x2x1_0_ttn_align8"));



}

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    '''),
  ],
  '100a': [
    KernelInstance(''' // 1
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_4,cute::_4,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_4,cute::_4,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_4x4x1_0_nnn_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 2

using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x1x1_0_nnn_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 3
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x2x1_0_nnn_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 4
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_4,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_4,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm
using cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_64x128x64_1x4x1_0_nnn_align8_1sm"));



}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 5
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_4, cute::_4, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm,

    cutlass::epilogue::fusion::LinearCombination<
      cutlass::bfloat16_t,
      float,
      cutlass::bfloat16_t,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_4, cute::_4, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_epilogue,
    void>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_4x4x1_0_ttt_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 6
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x1x1_0_ttt_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 7
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x2x1_0_ttt_align8_1sm"));


}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      '''),
    KernelInstance(''' // 8
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_4,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_64, cute::_128, cute::_64>,
    cute::Shape<cute::_1,cute::_4,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_epilogue::SharedStorage))>,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  >::CollectiveOp;

// Gemm operator cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm
using cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_mainloop,
    cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_epilogue>;

// Define named type
struct cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm :
  public cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm(Manifest &manifest) {



  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm>;
  manifest.append(
    new GemmUniversal3xOperation<GemmKernel>("cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_64x128x64_1x4x1_0_ttt_align8_1sm"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
      ''')
  ]
}

CommonIncludes = '''
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "library_internal.h"
#include "gemm_operation.h"
#include "gemm_operation_3x.hpp"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
'''

KernelModules = defaultdict(dict)

###############################################################################################################

#
def Clone(options):
  if not os.path.exists(options.cutlass_path):
    cmd = 'git clone --branch %s %s %s' % ( options.cutlass_branch, options.cutlass_git, options.cutlass_path)
    print("#", cmd)
    result = subprocess.call(cmd, shell=True)
  else:
    print("# CUTLASS exists at %s. Skipping CUTLASS checkout." % options.cutlass_path)
  if result != 0:
    return False

  return True

###############################################################################################################

#
def Setup(options):
  result = 0

  print(" # Setup")

  if not os.path.exists(options.src_dir):
    os.mkdir(options.src_dir)

  if not os.path.exists(options.tmp_dir):
    os.mkdir(options.tmp_dir)

  if not os.path.exists(options.results_dir):
    os.mkdir(options.results_dir)

  return True

###############################################################################################################

def SourceFileName(options, arch, kernels, index):
  return os.path.join(options.src_dir, "cutlass_%s_%s_%d_%d.cu" % (options.node, arch, len(kernels), index))

def ObjectFileName(options, arch, kernels, index):
  return os.path.join(options.tmp_dir, "cutlass_%s_%s_%d_%d.cu.o" % (options.node, arch, len(kernels), index))

def ProfileRawName(options, arch, kernels, build_ver, index):
  return os.path.join(options.tmp_dir, "cutlass_%s_%s_%d_%s_%d_profile_raw.csv" % (options.node, arch, len(kernels), build_ver, index))

def ProfileName(options, arch, kernels, build_ver, index):
  return os.path.join(options.results_dir, "cutlass_%s_%s_%d_%s_%d_profile.csv" % (options.node, arch, len(kernels), build_ver, index))

def BuildVersion(nvcc_path):
  nvcc_ver_output = os.popen(nvcc_path + ' --version').read().split('\n')
  output = nvcc_ver_output[4].split('.')
  return output[1] + "_" + output[2]

def TimeStamp():
  return date.today().strftime("%m-%d-%Y")

def NvccVersion(nvcc_path):
  nvcc_ver_output = os.popen(nvcc_path + ' --version').read().split('\n')
  return nvcc_ver_output[3]

def ModuleFilter(powerset, kernel_count):
  step = 1
  if kernel_count == 2:
    step = 7
  elif kernel_count == 4:
    step = 30

  indices = []
  filtered_kernels = []
  for index, kernel in enumerate(powerset):
    if index % step == 0:
      indices.append(index)
      filtered_kernels.append(kernel)
  return (filtered_kernels, indices)

###############################################################################################################

RawKeyString = 'source file name , phase name , phase input files , phase output file , arch , tool, metric , unit'
RenamedKeys = [x.lstrip().rstrip() for x in RawKeyString.split(',')]

RawKeys = keys = [x for x in RawKeyString.split(',')]
NewKeys = ['Date', 'Node', 'NVCC', 'Arch', 'Kernel Count', 'Combination Index']
Keys = NewKeys + RenamedKeys

###############################################################################################################

#
def EmitSourceFile(options, arch, kernels, index):
  source_file_name = SourceFileName(options, arch, kernels, index)
  with open(source_file_name, 'w') as source_file:
    source_file.write(CommonIncludes)
    for kernel in kernels:
      source_file.write('\n')
      source_file.write(kernel.source)
      source_file.write("\n\n")
    source_file.write('\n')

#
def ProfileSource(options, arch, kernels, nvcc_path, index):
  source_file_name = SourceFileName(options, arch, kernels, index)

  build_ver = BuildVersion(nvcc_path)
  object_file_name = ObjectFileName(options, arch, kernels, index)
  profile_raw_name = ProfileRawName(options, arch, kernels, build_ver, index)

  if os.path.exists(profile_raw_name):
    os.remove(profile_raw_name)

  cuda_bin_path = os.path.split(nvcc_path)[0]
  ptxDescPath = os.path.join(cuda_bin_path, 'ptxExtDesc.txt')

  nvcc_archs = '--generate-code=arch=compute_%s,code=[sm_%s]' % (arch, arch)

  nvcc_arguments = ' '.join([
    '-I%s' % os.path.join(options.cutlass_path, 'include'),
    '-I%s' % os.path.join(options.cutlass_path, 'tools/library/include'),
    '-I%s' % os.path.join(options.cutlass_path, 'tools/library/src'),
      "-forward-unknown-to-host-compiler",
      "-O3",
      "-DNDEBUG",
      "-Xcompiler=-fPIC",
      "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr ",
      "-DCUTLASS_TEST_LEVEL=0 ",
      "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 ",
      "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 ",
      "-DCUTLASS_DEBUG_TRACE_LEVEL=0 ",
      "-DCUTLASS_ENABLE_EXTENDED_PTX=1 ",
      "-DCUTLASS_ENABLE_INTERNAL_NVVM=1 ",
      "-DCUTLASS_CUDA_INTERNAL_L2_PREFETCH_ENABLED=1 ",
      "-DCUTLASS_CUDA_RP2RP_ENABLED=1 ",
      "-DCUTE_USE_PUBLIC_TMA_DESCRIPTOR=1 ",
      "-DCUTLASS_ENABLE_COMPILER_KNOBS=1 ",
      "-DCUTLASS_EMIT_KERNEL_METADATA_DIR=\\\"%s\\\" " % os.path.join(options.cutlass_path, 'build/tools/library/generated/metadata'),
      "-DCUDA_PTX_KNOB_SCHED_MEM_NO_ALIAS_ENABLED=1 ",
      "-DCUDA_PTX_KNOB_DISABLE_IMPLICIT_MEM_DESC_ENABLED=1 ",
      "-DCUDA_PTX_KNOB_COPYPROP_NOWRITENONRR_ENABLED=1 ",
      "-DCUDA_PTX_KNOB_DISABLE_WAR_ENABLED=1 ",
      "-Xcicc \"--Xllc -remat-loop-trip=500\" ",
      "-DJETFIRE_ENABLED=1 ",
      "-DCUTLASS_USE_INTERNAL_TMA_DESC=1 ",
      "-uumn",
      "-Xcompiler=-std=c++17",
      "-Xptxas=--ext-desc-file=%s" % ptxDescPath,
      "-Xcompiler=-Wconversion",
      "-Xcompiler=-fno-strict-aliasing",
      "-std=c++17",
      "-Xcicc \"--Xllc -remat-loop-trip=500\"",
    ])
  extra = [
    ]

  cmd = "%s %s -x cu -c %s -o %s %s --time %s" % (nvcc_path, nvcc_archs, source_file_name, object_file_name, nvcc_arguments, profile_raw_name)
  print("#", cmd)
  result = subprocess.call(cmd, shell=True)
  if result != 0:
    print("Error!")
    return False

  return True

#
def ProcessSource(options, arch, kernels, build_ver, nvcc_ver, index):

  profile_raw_name = ProfileRawName(options, arch, kernels, build_ver, index)
  profile_name = ProfileName(options, arch, kernels, build_ver, index)

  with open(profile_raw_name, 'r') as profile_raw_file:
    profile_raw = csv.DictReader(profile_raw_file)
    with open(profile_name, 'w', newline='') as profile_file:
      profile = csv.DictWriter(profile_file, fieldnames = Keys)
      profile.writeheader()

      for row_raw in profile_raw:

        row = {}
        row['Date'] = str(TimeStamp())
        row['Node'] = str(options.node)
        row['NVCC'] = str(nvcc_ver)
        row['Arch'] = str(arch)
        row['Kernel Count'] = str(len(kernels))
        row['Combination Index'] = str(index)
        for raw_key, renamed_key in zip(RawKeys, RenamedKeys):
          row[renamed_key] = str(row_raw[raw_key])

        profile.writerow(row)
  pass

#
def ProfileAll(options):

  # Emit all source files first to enable reusing same files with different compilers
  # For all kernel counts, emit all possible combinations of kernels
  for arch in options.architectures:
    for kernel_count in KernelCounts:
      kernel_list = KernelModules[arch][kernel_count][0]
      indices = KernelModules[arch][kernel_count][1]
      for kernel, index in zip(kernel_list, indices):
        EmitSourceFile(options, arch, kernel, index)

  for nvcc_path in options.nvcc_path_list:
    print( " Profiling with nvcc from path=%s" %(nvcc_path))
    for arch in options.architectures:
      for kernel_count in KernelCounts:
        kernel_list = KernelModules[arch][kernel_count][0]
        indices = KernelModules[arch][kernel_count][1]
        for kernel, index in zip(kernel_list, indices):
          result = ProfileSource(options, arch, kernel, nvcc_path, index)
          if not result:
            return False

          build_ver = BuildVersion(nvcc_path)
          nvcc_ver = NvccVersion(nvcc_path)
          ProcessSource(options, arch, kernel, build_ver, nvcc_ver, index)

  return True

###############################################################################################################

#
def AnalyzeAll(options):
  # concatenates all results in the results/ directory

  data_frames = []

  for concat_with in options.concatenate_with:
    if len(concat_with) > 0:
      print(" .. concatenating with", concat_with)
      data_frames.append(pandas.read_csv(concat_with))


  for nvcc_path in options.nvcc_path_list:
    print( " Analyze with nvcc from path=%s" %(nvcc_path))
    for arch in options.architectures:
      for kernel_count in KernelCounts:
        kernel_list = KernelModules[arch][kernel_count][0]
        indices = KernelModules[arch][kernel_count][1]
        for kernel, index in zip(kernel_list, indices):
          build_ver = BuildVersion(nvcc_path)
          profile_name = ProfileName(options, arch, kernel, build_ver, index)
          data_frames.append(pandas.read_csv(profile_name))

  print("Concatenating %d data frames" % len(data_frames))

  final_df = pandas.concat(data_frames, ignore_index=True)
  final_df.to_csv(options.final_result)

###############################################################################################################

#
def Main():
  options = Options()
  for arch in options.architectures:
    kernel_arch_list = Kernels[arch]
    for kernel_count in KernelCounts:
      powerset = []
      for kernel_combinations in itertools.combinations(kernel_arch_list, kernel_count):
        powerset.append(kernel_combinations)
      KernelModules[arch][kernel_count] = ModuleFilter(powerset, kernel_count)

  phases = {
    'clone': Clone,
    'setup': Setup,
    'profile': ProfileAll,
    'analyze': AnalyzeAll
  }
  for phase in options.phases:
    print("phase: %s" % phase)
    result = phases[phase](options)
    if not result:
      return -1
  return 0

###############################################################################################################

#
if __name__ == '__main__':
  sys.exit(Main())

###############################################################################################################
