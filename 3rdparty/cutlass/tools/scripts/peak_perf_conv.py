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
from conv_network import Resnet50, VGG_16, VNet, DarpaNet, StridedLayers, LargeStridedLayers
import re, pdb
#################################################################################################

DataType = [
  # HMMA
  {
    'name': 'F16 <= F16*F16 + F32',
    'tag': 'TensorOp(F16)',
    'identifier': 'hmma_f32_f16',
    'input':  '--Activation=f16',
    'filter': '--Filter=f16',
    'output': '--Output=f16',
    'accum': '--accum=f32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'F16 <= F16*F16 + F16',
    'tag': 'Inference(F16)',
    'identifier': 'hmma_f16_f16',
    'input':  '--Activation=f16',
    'filter': '--Filter=f16',
    'output': '--Output=f16',
    'accum': '--accum=f16',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'F32 <= TF32*TF32 + F32',
    'tag': 'TensorOp(F32)',
    'identifier': 'hmma_f32_tf32',
    'input':  '--Activation=f32',
    'filter': '--Filter=f32',
    'output': '--Output=f32',
    'accum': '--accum=f32',
    'opclass': '--op_class=tensorop'
  },
  # IMMA
  {
    'name': 'S32 <= S8*S8 + S32 (nhwc)',
    'tag': 'Inference(S8)[nhwc]',
    'identifier': 'imma_s32_s8_nhwc',
    'input':  '--Activation=s8:nhwc',
    'filter': '--Filter=s8:nhwc',
    'output': '--Output=s32:nhwc',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S32 <= S8*S8 + S32 (nc32hw32)',
    'tag': 'Inference(S8)[nc32hw32]',
    'identifier': 'imma_s32_s8_nc32hw32',
    'input':  '--Activation=s8:nc32hw32',
    'filter': '--Filter=s8:nc32hw32',
    'output': '--Output=s32:nc32hw32',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S8 <= S8*S8 + S32 (nhwc)',
    'tag': 'Inference(S8)[nhwc]',
    'identifier': 'imma_s8_s8_nhwc',
    'input':  '--Activation=s8:nhwc',
    'filter': '--Filter=s8:nhwc',
    'output': '--Output=s8:nhwc',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S8 <= S8*S8 + S32 (nc32hw32)',
    'tag': 'Inference(S8)[nc32hw32]',
    'identifier': 'imma_s8_s8_nc32hw32',
    'input':  '--Activation=s8:nc32hw32',
    'filter': '--Filter=s8:c32rsk32',
    'output': '--Output=s8:nc32hw32',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S32 <= S4*S4 + S32',
    'tag': 'Inference(S4)[nhwc]',
    'identifier': 'imma_s32_s4_nhwc',
    'input':  '--Activation=s4:nhwc',
    'filter': '--Filter=s4:nhwc',
    'output': '--Output=s32:nhwc',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S4 <= S4*S4 + S32 (nhwc)',
    'tag': 'Inference(S4)[nhwc]',
    'identifier': 'imma_s4_s4_nhwc',
    'input':  '--Activation=s4:nhwc',
    'filter': '--Filter=s4:nhwc',
    'output': '--Output=s4:nhwc',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  {
    'name': 'S4 <= S4*S4 + S32 (nc64hw64)',
    'tag': 'Inference(S4)[nc64hwc64]',
    'identifier': 'imma_s4_s4_nc64hw64',
    'input':  '--Activation=s4:nc64hw64',
    'filter': '--Filter=s4:c64rsk64',
    'output': '--Output=s4:nc64hw64',
    'accum': '--accum=s32',
    'opclass': '--op_class=tensorop'
  },
  # FMMA
  {
    'name': 'F32 <= F32*F32 + F32',
    'tag': 'FFMA(F32)',
    'identifier': 'ffma',
    'input':  '--Activation=f32',
    'filter': '--Filter=f32',
    'output': '--Output=f32',
    'accum': '--accum=f32',
    'opclass': '--op_class=simt'
  },
  {
    'name': 'CF32 * CF32 + CF32',
    'tag': 'FFMA(CF32)',
    'identifier': 'cf32_cf32_cf32',
    'input':  '--Activation=cf32',
    'filter': '--Filter=cf32',
    'output': '--Output=cf32',
    'accum': '--accum=cf32',
  },
]
## TODO - add more templates above for mixed complex operations as need be
# convolutional networks defined in conv_network.py

ConvolutionalNetworkList = [Resnet50, VGG_16,  VNet, DarpaNet, StridedLayers, LargeStridedLayers]

ConvolutionNetworks = dict([(x.name, x) for x in ConvolutionalNetworkList])

#################################################################################################

def ResultName(args, provider, data_type, network):
  return "%s_%s_%s_%s_%s_%s.csv" % (provider, args.operation, args.conv_kind, args.chip, data_type['identifier'], network)

#################################################################################################

def ProfileLayer(args, data_type, layer):

  tmp_output_name = ResultName(args, args.providers, data_type, layer.network)
  
  nv_nsight = False
  if args.nv_nsight_path != "Unknown":
    nv_nsight = True

  # add a tag for strided convolution
  stride = "unity stride"
  if (layer.is_strided()):
    stride = "strided"

  cmdline_base = args.profiler_path
  cmdline_base += " --tags=\"chip:%s,clock:%s,Network:%s,layer_id:%s,stride:%s,%s\"" % (args.chip, args.clock, layer.network, layer.id, stride, args.tags)
  cmdline_base += " --device=%s" % args.device
  cmdline_base += " --clock=%s" % args.clock
  cmdline_base += " --profiling-iterations=%s" % (args.profiling_iterations if nv_nsight == False else "1")
  cmdline_base += " --operation=%s  --conv_kind=%s --providers=%s" % (args.operation, args.conv_kind, args.providers)
  cmdline_base += " --verification-providers=%s" % args.verification_providers
  cmdline_base += " --eq-gemm-provider=%s" % args.eq_gemm_provider
  cmdline_base += " --split-k-slices=%s" % args.split_k_slices
  cmdline_base += " --split-k-mode=%s" % args.split_k_mode
  cmdline_base += " --output=%s" % tmp_output_name
  cmdline_base += " --append=%s" % args.append.lower()

  if nv_nsight:
    cmdline_base = args.nv_nsight_path + "/nv-nsight-cu-cli " + cmdline_base
    cmdline_base += " --warmup-iterations=%s" % "0"

  # add kernel regex 
  kernels_base = args.kernels
  if args.providers == 'cutlass' and\
   args.conv_kind == 'dgrad' and\
   layer.params['stride_h'] == 1 and\
   layer.params['stride_w'] == 1:

    kernels_base += ",%s" % "unity_stride"

  cmdline_base += " --kernels=%s" % kernels_base

  cmdline_data_type = " ".join([data_type['input'], data_type['filter'], data_type['output'], data_type['accum'], data_type['opclass']])

  cmdline_layer = layer.profiler_cmd(int(args.batch_size))

  cmdline = " ".join([cmdline_base, cmdline_data_type, cmdline_layer])

  print("\"%s\"" % cmdline)

  return_code = subprocess.call(cmdline, shell=True)

  if return_code:
    print("Error:\n")
    print(cmdline)

  return return_code

#################################################################################################
#                                   Profiling functions
#################################################################################################
def ProfileNetwork(args, data_type, network):
  for layer in network.layers:
    ProfileLayer(args, data_type, layer)

#################################################################################################

def Profile(args):

  if args.network_name not in ConvolutionNetworks:
    print('Undefined convolution network!!! \"%s\". Choose a convolutional network from from %s' % (args.network_name, ConvolutionNetworks.keys()))
    sys.exit(0)

  for data_type in DataType:
    # profile only user requested math instructions
    if args.math_instruction != "all" and \
        data_type['identifier'] not in args.math_instruction.split(","):
      continue

    ProfileNetwork(args, data_type, ConvolutionNetworks[args.network_name])

  pass
#################################################################################################

#################################################################################################
#                                  Processing functions
#################################################################################################
def ProcessResultFile(args, provider, data_type, layer):

  src_file_name = ResultName(args, provider, data_type, layer.network)

  filename, file_extension = os.path.splitext(src_file_name)
  conv_filename = "".join([filename, '.', args.operation, file_extension])

  peak_gflops = -1

  top_conv_row = {}
  top_eq_gemm_row = {}
  eq_gemm_op_cache = {}
  batch_size = int(args.batch_size)

  with open(conv_filename, 'r') as src_file:
    reader = csv.DictReader(src_file)

    for row in reader:
      # result csv file may have many entries appended into one file
      # process layer for a specific layer_id and batch_size 

      layer_id = int(row['layer_id'])
      layer_batch_size = int(row['n'])

      if layer_id != layer.id or layer_batch_size != batch_size:
      #if layer_id != layer.id: # use this for layers running different batch sizes
        continue

      # cache results for equivalent gemm operation and continue
      if row['OperationKind'] == 'eq_gemm':
        eq_gemm_op_cache[row['Operation']] = row
        continue

      # skip layers not supported with a specific provider
      if row['Status'] != 'success':
        continue

      gflops = float(row['GFLOPs'])

      if peak_gflops < 0 or peak_gflops < gflops:
        peak_gflops = gflops
        top_conv_row = row

  # find equivalent gemm operation profiling results equal to that of top_conv_row['Operation']
  if top_conv_row and (top_conv_row['Operation'] in eq_gemm_op_cache):
    top_eq_gemm_row = eq_gemm_op_cache[top_conv_row['Operation']]

  return top_conv_row, top_eq_gemm_row

#################################################################################################
#               Headers in processed final csv file containing top operations
#################################################################################################
def CreateCSVFieldNames(args):
  tagnames = []
  
  # tagnames for user-defined tags
  if args.tags != '':
    for tag in args.tags.split(','):
      tagnames.append(tag.split(':')[0])

  # publicfields for public performance data    
  rownames = ['Network', 'DataTypeOp', 'DataTypeTag', 'chip', 'conv_kind', 'layer_id', 'Provider', \
                'OperationKind', 'Operation', 'BatchSize', 'LayerParams', 'stride', 'iterator_algorithm',\
                'split_k_mode', 'split_k_slices', 'Runtime', 'GFLOPs', 'Speedup']
  
  # internalfields for internal performance data
  if args.release_type == 'internal':
    rownames.append('MathUtilization')

  return rownames, tagnames


def CreateCSVEntry(args, provider, data_type, layer, top_op_row):

  rownames, tagnames = CreateCSVFieldNames(args)
  fieldnames = tagnames + rownames
  csv_row = {}

  # create csv row with data (fieldname as key, data as value)
  for fieldname in fieldnames:
    if fieldname == 'DataTypeOp':
      csv_row[fieldname] = data_type['name']
    elif fieldname == 'DataTypeTag':
      csv_row[fieldname] = data_type['tag']
    elif fieldname == 'BatchSize':
      csv_row[fieldname] = args.batch_size
    elif fieldname == 'LayerParams':
      csv_row[fieldname] = layer.profiler_cmd(int(args.batch_size))
    elif fieldname == 'Speedup':
      csv_row[fieldname] = 0
    else: # fieldname found directly in profiler generated csv
      csv_row[fieldname] = str(top_op_row[fieldname])

  return csv_row

#################################################################################################
#     Emit command lines for top operations (helpful for reruns without full sweeps)
#################################################################################################
def  EmitCommandLinesTopOperations(args, results):

  # command line file name 
  cmd_file_name = os.path.splitext(args.output)[0]+'.sh'

  # open file in append mode
  with open(cmd_file_name, 'a', newline='') as cmd_file:
    for result in results:
      #print(result)

      data_type = next((item for item in DataType if item["name"] == result['DataTypeOp']), None)
      if data_type==None:
        print("DataType not found in dictionary")
        pass


      cmdline = args.profiler_path
      cmdline += " --tags=Network:%s,DataTypeTag:\"%s\",chip:%s,clock:%s,layer_id:%s,BatchSize:%s,stride:%s,%s" %\
                   (result['Network'] ,data_type['tag'], args.chip, args.clock, result['layer_id'], result['BatchSize'], result['stride'], args.tags)
      cmdline += " --clock=%s" % args.clock
      cmdline += " --operation=%s" % (args.operation)
      cmdline += " --kernels=%s" % result['Operation']
      cmdline += " %s" % result['LayerParams']
      cmdline += " --split-k-slices=%s" % result['split_k_slices']
      cmdline += " --split-k-mode=%s" % result['split_k_mode']
      cmdline += " --output=%s" % 'top_operations.csv'
      cmdline += " --append=%s" % args.append.lower()
      cmdline += " --device=%s\n" % args.device

      cmd_file.writelines(cmdline)
  return

#################################################################################################
#  Process a network sweep to find top operations and create two files:
#   1. A csv file with top operations + {split-k-mode, split-k-slices} for each (layer, provider)
#   2. A sh file with commandlines for top operations + {split-k-mode, split-k-slices} 
#################################################################################################
def ProcessNetwork(args, network):

  rownames, tagnames = CreateCSVFieldNames(args)
  fieldnames = tagnames + rownames

  open_type = 'a' if args.append.lower() == "true" or args.append.lower() == "1" else 'w'
  file_exists = os.path.isfile(args.output)

  with open(args.output, open_type, newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    
    if not file_exists:
      writer.writeheader()

    for data_type in DataType:

      # process only user requested math instructions
      if args.math_instruction != "all" and \
          data_type['identifier'] not in args.math_instruction.split(","):
        continue

      # make a pass over all layers
      for layer in network.layers:
        results = []

        baseline_gflops = 0

        # make a pass over all providers
        for provider in args.providers.split(','):

          top_conv_row, top_eq_gemm_row = ProcessResultFile(args, provider, data_type, layer)

          if provider == args.baseline_provider:
            baseline_gflops = top_conv_row['GFLOPs']

          # create and append top_conv_row to the processed file
          if top_conv_row:
            results.append(CreateCSVEntry(args, provider, data_type, layer, top_conv_row))

          # if equivalent gemm is profiled create and append top_eq_gemm_row to the processed file
          if top_eq_gemm_row:
            results.append(CreateCSVEntry(args, provider, data_type, layer, top_eq_gemm_row))

        # compute speedup
        if baseline_gflops:
          for result in results:
            result['Speedup'] = float(result['GFLOPs']) / baseline_gflops

        # emit top operation into .csv (for plotting) and .sh (for re-runs)
        for result in results:
          writer.writerow(result)
          if provider == 'cutlass':
            EmitCommandLinesTopOperations(args, results)

  pass

#
def Process(args):

  if args.network_name not in ConvolutionNetworks:
    print('Undefined convolution network!!! \"%s\". Choose a convolutional network from from %s' % (args.network_name, ConvolutionNetworks.keys()))
    sys.exit(0)

  ProcessNetwork(args, ConvolutionNetworks[args.network_name])
    
  pass
#################################################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--tags", default="", help="Additions to pivot tags to prepend to table")
  parser.add_argument("--chip", default="Unknown", help="Specifies the name of the chip")
  parser.add_argument("--device", default="0", help="Device ID to run cutlass profiler")
  parser.add_argument("--clock", default="", help="SM clock speed in MHz")
  parser.add_argument("--nv-nsight-path", default="Unknown", help="Path to nv-nsight-cu-cli binary")
  parser.add_argument("--profiling-iterations", default="10", help="Number of iterations to profile each kernel")
  parser.add_argument("--network-name", default="Resnet50", help="Convolution network to profile. Choose from (Resnet50, VNet, StridedLayers, LargeStridedLayers)")
  parser.add_argument("--operation", default="conv2d", help="cutlass convolution operation/dimension. Choose from (conv2d/conv3d)")
  parser.add_argument("--kernels", default="cutlass", help="cutlass appears in all kernels")
  parser.add_argument("--sm-count", default="68", help="Number of SMs within the machine")
  parser.add_argument("--math-instruction", default="all", help="Math instruction to profile (hmma_f32_f16, hmma_f32_tf32, imma_s32_s8_nhwc, imma_s32_s8_nc32hw32, ffma)")
  parser.add_argument("--batch-size", default="-1", help="Number input images in one convolution")
  parser.add_argument("--conv-kind", default="fprop", help="Convolution operator (fprop, dgrad, wgrad)")
  parser.add_argument("--providers", default="cutlass", help="Provider to measure")
  parser.add_argument("--verification-providers", default="unknown", help="Verification provider to validate cutlass resutls")
  parser.add_argument("--eq-gemm-provider", default="none", help="Profiles implicit gemm's equivalent gemm version")
  parser.add_argument("--baseline-provider", default="cublas", help="Baseline provider")
  parser.add_argument("--split-k-mode", default="serial", help="Sweep conv2d operator with split-k-mode")
  parser.add_argument("--split-k-slices", default="1", help="Sweep conv operator with split-k-slices")
  parser.add_argument("--profiler-path", default='./tools/profiler/cutlass_profiler', help="Path to CUTLASS Profiler binary to run")
  parser.add_argument("--phases", default="profile,process", help="Phases of profiling to construct")
  parser.add_argument("--output", default="cutlass_conv_performance_result.csv", help="Name of output file")
  parser.add_argument("--release-type", default="internal", help="If 'internal', then internal-only quantities like MathUtilization are included in processed output.")
  parser.add_argument("--append", default="true", help="If true, final result file is opened for append.")

  args = parser.parse_args()
  
  for phase in args.phases.split(','):
    if phase == 'profile':
      Profile(args)
    elif phase == 'process':
      Process(args)

#################################################################################################
