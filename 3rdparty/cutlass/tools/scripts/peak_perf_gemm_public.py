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

import sys
import csv
import os
import subprocess
import argparse

#################################################################################################

#
# Workloads are based on SM count to perfectly quantize
#

WorkloadsOuterLarge = {
  48: '--m=2048,4096 --n=3072,4608',
  72: '--m=2048,4096 --n=2304,4608',
  80: '--m=2048,4096 --n=2560,5120',
  84: '--m=2688,5376 --n=4096',
  108: '--m=3456,6912 --n=4096',
  132: '--m=2048,4096 --n=4224',
  142: '--m=9088 --n=4096',
  144: '--m=2048,4096 --n=4608',
}

WorkloadsOuterSmall = {
  48: '--m=1536 --n=3072',
  72: '--m=2048 --n=2304',
  80: '--m=2048 --n=2560',
  84: '--m=2688 --n=2048',
  108: '--m=3456 --n=4096',
  132: '--m=2048 --n=4224',
  142: '--m=4544 --n=4096',
  144: '--m=2048 --n=4608',
}

WorkloadsInnerLarge = "--k=1024,2048,4096,8192,10240,16384"
WorkloadsInnerMedium = "--k=1024,2048,4096,6144"
WorkloadsInnerSmall = "--k=768"

#################################################################################################

Groups = [
  {
    'name': 'SGEMM',
    'identifier': 'sgemm',
    'workloads_outer': WorkloadsOuterSmall,
    'workloads_inner': WorkloadsInnerMedium,
    'kernels': {
      'tu102': 'sgemm*128x128',
      'gv100': 'sgemm*128x128',
      'ga100': 'sgemm*128x128',
      'ga102': 'sgemm*128x128',
      'ga107': 'sgemm*128x128',
      'gh100': 'sgemm*128x128',
      'ad102': 'sgemm*128x128',
    },
    'filter': '--accum=f32 --opcode-class=simt',
    'layouts': [
      ('NN', '--A=f32:column --B=f32:column'),
      ('NT', '--A=f32:column --B=f32:row'),
      ('TN', '--A=f32:row --B=f32:column'),
      ('TT', '--A=f32:row --B=f32:row')
    ]
  },
  {
    'name': 'DGEMM',
    'identifier': 'dgemm',
    'kernels': {
      'tu102': 'dgemm',
      'gv100': 'dgemm',
      'ga100': 'd88*gemm',
      'ga102': 'cutlass_tensorop_d884gemm_64x64_16x4',
      'ga107': 'd88*gemm',
      'gh100': 'd168*gemm',
      'ad102': 'cutlass_tensorop_d884gemm_64x64_16x4',
    },
    'workloads': {
      'tu102': {'outer': WorkloadsOuterSmall, 'inner': WorkloadsInnerSmall},
      'gv100': {'outer': WorkloadsOuterLarge, 'inner': WorkloadsInnerMedium},
      'ga100': {'outer': WorkloadsOuterLarge, 'inner': WorkloadsInnerMedium},
      'ga102': {'outer': WorkloadsOuterSmall, 'inner': WorkloadsInnerSmall},
      'ga107': {'outer': WorkloadsOuterSmall, 'inner': WorkloadsInnerSmall},
      'gh100': {'outer': WorkloadsOuterLarge, 'inner': WorkloadsInnerMedium},
			'ad102': {'outer': WorkloadsOuterSmall, 'inner': WorkloadsInnerSmall},
    },
    'filter': '--accum=f64',
    'layouts': [
      ('NN', '--A=f64:column --B=f64:column'),
      ('NT', '--A=f64:column --B=f64:row'),
      ('TN', '--A=f64:row --B=f64:column'),
      ('TT', '--A=f64:row --B=f64:row')
    ]
  },
  {
    'name': 'TensorOp (f16)',
    'identifier': 'tensorop_f16',
    'kernels': {
      'tu102': 'tensorop_h1688gemm*128x128*align8',
      'gv100': 'tensorop_h884gemm*128x128*align8',
      'ga100': 'tensorop_h16816gemm*128x128*align8,tensorop_h16816gemm*256*align8',
      'ga102': 'tensorop_h16816gemm*128x128*align8,tensorop_h16816gemm*256*align8',
      'ga107': 'tensorop_h16816gemm*128x128*align8,tensorop_h16816gemm*256*align8',
      'ad102': 'tensorop_h16816gemm*128x128*align8,tensorop_h16816gemm*256*align8',
      'gh100': 'cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16*align8',
    },
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'filter': '--accum=f16 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
  {
    'name': 'TensorOp (f32)',
    'identifier': 'tensorop_f32',
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'kernels': {
      'tu102': 'tensorop_f16_s1688gemm*128x128*align8',
      'gv100': 'tensorop_f16_s884gemm*128x128*align8',
      'ga100': 'tensorop_f16_s16816gemm*128x128*align8,tensorop_fp16_s16816gemm*256*align8',
      'ga102': 'tensorop_f16_s16816gemm*128x128*align8,tensorop_fp16_s16816gemm*256*align8',
      'ga107': 'tensorop_f16_s16816gemm*128x128*align8,tensorop_fp16_s16816gemm*256*align8',
      'ad102': 'tensorop_f16_s16816gemm*128x128*align8,tensorop_fp16_s16816gemm*256*align8',
      'gh100': 'cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16*align8'
    },
    'filter': '--accum=f32 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
  {
    'name': 'TensorOp (TF32)',
    'identifier': 'tensorop_tf32',
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'kernels': {
      'tu102': '',
      'gv100': '',
      'ga100': 'tensorop_s1688tf32gemm*128x128*align4,tensorop_s1688tf32gemm*256*align4',
      'ga102': 'tensorop_s1688tf32gemm*128x128*align4,tensorop_s1688tf32gemm*256*align4',
      'ga107': 'tensorop_s1688tf32gemm*128x128*align4,tensorop_s1688tf32gemm*256*align4',
      'ad102': 'tensorop_s1688tf32gemm*128x128*align4,tensorop_s1688tf32gemm*256*align4',
      'gh100': 'cutlass3x_sm90_tensorop_tf32gemm*f32_f32_f32_f32*align4'
    },
    'filter': '--accum=f32 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f32:column --B=f32:column'),
      ('NT', '--A=f32:column --B=f32:row'),
      ('TN', '--A=f32:row --B=f32:column'),
      ('TT', '--A=f32:row --B=f32:row')
    ]
  },
]

# rename this if you want to measure IDP4A perf
GroupsIdp4a = [
  {
    'name': 'TensorOp (3 x TF32)',
    'identifier': 'tensorop_3xtf32',
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'kernels': {
      'tu102': '',
      'gv100': '',
      'ga100': 'tensorop_s1688gemm_128x128*align4,tensorop_s1688gemm_*256*align4',
    },
    'filter': '--accum=f32 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f32:column --B=f32:column'),
      ('NT', '--A=f32:column --B=f32:row'),
      ('TN', '--A=f32:row --B=f32:column'),
      ('TT', '--A=f32:row --B=f32:row')
    ]
  },
  {
    'name': 'TensorOp (int8)',
    'identifier': 'tensorop_s8',
    'kernels': {
      'tu102': 'cutlass_tensorop_s8_i8816gemm_s8_*_align16',
      'gv100': 'igemm*128x128',
      'ga100': 'cutlass_tensorop_s8_i16832gemm_s8_*_align16',
    },
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'filter': '--accum=s32',
    'layouts': [
      ('NT', '--A=s8:nk32 --B=s8:tk32'),
      ('TN', '--A=s8:row --B=s8:column'),
    ]
  },
]

# rename this if you want to measure WMMA perf
GroupsWmma = [
  {
    'name': 'TensorOp (f16)',
    'identifier': 'tensorop_f16',
    'kernels': {
      'tu102': 'tensorop_h1688gemm*128x128*align8',
      'gv100': 'tensorop_h884gemm*128x128*align8',
      'ga100': 'tensorop_h16816gemm*128x128*align8,tensorop_h16816gemm*256*align8',
    },
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'filter': '--accum=f16 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
  {
    'name': 'WMMA (f16)',
    'identifier': 'wmma_tensorop_f16',
    'kernels': {
      'tu102': 'wmma_tensorop_h*gemm*128x128*align8',
      'gv100': 'wmma_tensorop_h*gemm*128x128*align8',
      'ga100': 'wmma_tensorop_h*gemm*128x128*align8,wmma_tensorop_h*gemm*256*align8',
    },
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'filter': '--accum=f16 --opcode-class=wmmatensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
  {
    'name': 'TensorOp (f32)',
    'identifier': 'tensorop_f32',
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'kernels': {
      'tu102': 'tensorop_f16_s1688gemm*128x128*align8',
      'gv100': 'tensorop_f16_s884gemm*128x128*align8',
      'ga100': 'tensorop_f16_s16816gemm*128x128*align8,tensorop_s16816gemm*256*align8',
    },
    'filter': '--accum=f32 --opcode-class=tensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
  {
    'name': 'WMMA (f32)',
    'identifier': 'wmma_tensorop_f32',
    'workloads_outer': WorkloadsOuterLarge,
    'workloads_inner': WorkloadsInnerLarge,
    'kernels': {
      'tu102': 'wmma_tensorop_f16_s*gemm*128x128*align8',
      'gv100': 'wmma_tensorop_f16_s*gemm*128x128*align8',
      'ga100': 'wmma_tensorop_f16_s*gemm*128x128*align8,wmma_tensorop_s*gemm*256*align8',
    },
    'filter': '--accum=f32 --opcode-class=wmmatensorop',
    'layouts': [
      ('NN', '--A=f16:column --B=f16:column'),
      ('NT', '--A=f16:column --B=f16:row'),
      ('TN', '--A=f16:row --B=f16:column'),
      ('TT', '--A=f16:row --B=f16:row')
    ]
  },
]

#################################################################################################

def ResultName(chip, group, layout):
  return "%s_%s_%s" % (chip, group['identifier'], layout[0])

def ResultFileName(chip, group, layout):
  return ResultName(chip, group, layout) + ".gemm.csv"

#################################################################################################

def ProfileLayout(args, group, layout):

  tmp_output_name = ResultName(args.chip, group, layout)

  cmdline_base = args.profiler_path
  cmdline_base += " --tags=\"chip:%s,clock:%s%s\"" % (args.chip, args.clock, args.tags)
  cmdline_base += " --providers=%s" % args.providers
  cmdline_base += " --output=%s" % tmp_output_name
  cmdline_base += " --clock=%s" % args.clock
  cmdline_base += " --profiling-iterations=15"
  cmdline_base += " --operation=gemm"

  kernels = group['kernels'][args.chip]
  if kernels != '':
    kernels = ' --kernels=' + kernels

  cmdline_workload = ''
  
  # Select workloads based on data type, chip, and SM count
  if 'workloads_outer' in group.keys() and 'workloads_inner' in group.keys():
    workloads = (group['workloads_outer'], group['workloads_inner'])
  elif 'workloads' in group.keys():
    workloads = (group['workloads'][args.chip]['outer'], group['workloads'][args.chip]['inner'])

  workloads_str = "%s %s" % (workloads[0][int(args.sm_count)], workloads[1])

  # Command line
  try:
    cmdline_workload = group['filter'] + " " + kernels + " " + layout[1] + " " + workloads_str
  except:
    print(cmdline_base)

  cmdline = cmdline_base + " " + cmdline_workload

  print("\"%s\"" % cmdline)
  return_code = subprocess.call(cmdline, shell=True)

  if return_code:
    print("Error:\n")
    print(cmdline)

  return return_code
#

#################################################################################################

#
def Build(args):
  # maps SM count to architectures
  chips = {
    'tu102': '75',
    'gv100': '70',
    'ga100': '80',
    'ga102': '86',
    'ga107': '86',
    'ad102': '89',
    'gh100': '90a'
  }

  sm_arch = chips[args.chip]
  kernels = ','.join([group['kernels'][args.chip] for group in Groups if group['kernels'][args.chip] != ''])

  cmdline = 'cmake .. -DCUTLASS_ENABLE_CUDNN=OFF -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_NVCC_ARCHS=%s -DCUTLASS_LIBRARY_KERNELS=%s' % (sm_arch, kernels)
  print("CMake call:\n%s\n", cmdline)
  return_code = subprocess.call(cmdline, shell=True)
  
  if return_code:
    print('Error:\n')
    print(cmdline)
    return return_code

  cmdline = 'make cutlass_profiler -j12'
  return_code = subprocess.call(cmdline, shell=True)
  
  if return_code:
    print('Error:\n')
    print(cmdline)
    return return_code
  
  return 0
#

#################################################################################################

#
def Profile(args):

  for group in Groups:
    for layout in group['layouts']:
      ProfileLayout(args, group, layout)

  pass

#################################################################################################

#
def ProcessResultFile(args, provider, group, layout):
  '''
  Runs the workload and reports the peak gflops for each provider
  '''
  src_file_name = ResultFileName(args.chip, group, layout)

  peak_runtime = { 'cutlass': -1, 'cublas': -1}
  peak_gflops = {'cutlass': -1, 'cublas': -1}
  peak_utilization = {'cutlass': 0, 'cublas': 0}

  with open(src_file_name, 'r') as src_file:
    reader = csv.DictReader(src_file)

    for row in reader:
      try:
        provider = row['Provider'].lower()
        runtime = float(row['Runtime'])
        gflops = float(row['GFLOPs'])
        utilization = 0

        if args.release_type == 'internal':
          utilization = float(row['MathUtilization'])

        if peak_gflops[provider] < 0 or peak_gflops[provider] < gflops:
          peak_runtime[provider] = runtime
          peak_gflops[provider] = gflops
          peak_utilization[provider] = utilization
      except:
        pass

  return (peak_runtime, peak_gflops, peak_utilization)

#################################################################################################

def Process(args):

  fieldnames = ['Name', 'Provider','Chip','Product', 'Clock', 'Layout', 'Runtime', 'GFLOPs', 'Speedup']
  if args.release_type == 'internal':
    fieldnames += 'Utilization'

  open_type = 'a' if args.append.lower() == "true" or args.append.lower() == "1" else 'w'

  write_headers = True if not os.path.exists(args.output) or open_type != 'a' else False

  products = {
    'tu102': '2080Ti',
    'gv100': 'TitanV',
    'ga100': 'A100',
    'ga102': 'A40',
    'ga107': 'A2',
    'ad102': 'L40',
    'gh100': 'H100'
  }

  product_name = args.product if args.product != "" else products[args.chip]

  with open(args.output, open_type, newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)

    if write_headers:
      writer.writeheader()

    for group in Groups:
      for layout in group['layouts']:

        results = []

        baseline_gflops = 0

        # make a pass over all providers
        for provider in args.providers.split(','):

          runtime, gflops, utilization = ProcessResultFile(args, provider, group, layout)

          if provider == args.baseline_provider:
            baseline_gflops = gflops[args.baseline_provider]

          result_row = {
            'Name': group['name'],
            'Provider': provider,
            'Chip': args.chip,
            'Product': product_name,
            'Clock': args.clock,
            'Layout': layout[0],
            'Runtime': str(runtime[provider]),
            'GFLOPs': str(gflops[provider]),
            'Speedup': 0
          }

          if args.release_type == 'internal':
            result_row['Utilization'] = str(utilization[provider])

          results.append(result_row)

        # compute speedup
        if baseline_gflops:
          for result in results:
            result['Speedup'] = float(result['GFLOPs']) / baseline_gflops

        # emit if result is defined
        for result in results:
          if float(result['GFLOPs']) > 0:
            writer.writerow(result)

  pass

#################################################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--tags", default="", help="Additiona pivot tags to prepend to table")
  parser.add_argument("--chip", default="ga100", help="Specifies the name of the chip")
  parser.add_argument("--product", default="", help="Specifies the name of the product.")
  parser.add_argument("--clock", default="", help="SM clock speed in MHz")
  parser.add_argument("--sm-count", default="108", help="Number of SMs within the machine")
  parser.add_argument("--providers", default="cutlass,cublas", help="Provider to measure")
  parser.add_argument("--baseline-provider", default="cublas", help="Baseline provider")
  parser.add_argument("--profiler-path", default='./tools/profiler/cutlass_profiler', help="Path to CUTLASS Profiler binary to run")
  parser.add_argument("--phases", default="build,profile,process", help="Phases of profiling to construct: build,profile,process")
  parser.add_argument("--output", default="cutlass_performance_result.csv", help="Name of output file")
  parser.add_argument("--release-type", default="public", help="If 'internal', then internal-only quantities like MathUtilization are included in processed output.")
  parser.add_argument("--append", default="false", help="If true, final result file is opened for append.")

  args = parser.parse_args()

  for phase in args.phases.split(','):
    if phase == 'build':
      Build(args)
    if phase == 'profile':
      Profile(args)
    elif phase == 'process':
      Process(args)

#################################################################################################
