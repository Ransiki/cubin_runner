#!/usr/bin/python3
#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#################################################################################################

import argparse
import sys
from typing import *
from generator_library import *


def emit_op_selector_header(file):
  header = """\
// Given a 1x tmem copy op, returns the widest repeated variant that divides the specified bits in the N-mode
template <class CopyOp, int bits_n>
CUTE_HOST_DEVICE constexpr
auto
op_repeater()
{
"""
  file.write(header)

def emit_op_selector_body(file, op, dp, bits, dp_bits, repeats, pack16b, else_='else '):
  body = """\
  {else_}if constexpr (cute::is_same_v<CopyOp, SM100_{op}_{dp}dp{bits}b1x{op_16b}>) {{{clauses}
  }}
"""
  clause = """
    {else_}if constexpr (bits_n % ({dp_bits} * {_repeat}) == 0) {{
      return SM100_{op}_{dp}dp{bits}b{repeat}x{op_16b}{{}};
    }}\
"""
  op_16b = "_16b" if pack16b else ""

  clauses = ''
  for i, repeat in enumerate(reversed(repeats)):
    clauses += clause.format(op=op,
                             dp=dp,
                             bits=bits,
                             dp_bits=dp_bits,
                             repeat=repeat,
                             _repeat=str(repeat).rjust(len(str(repeats[-2]))),
                             op_16b=op_16b,
                             else_='' if i == 0 else 'else ')

  file.write(body.format(op=op,
                         dp=dp,
                         bits=bits,
                         repeat=repeat,
                         op_16b=op_16b,
                         else_=else_,
                         clauses=clauses))


def emit_op_selector_footer(file):
  footer = """\
  else {
    static_assert(dependent_false<CopyOp>, "Must pass 1x tmem copy operator");
  }
}

"""
  file.write(footer)


def emit_ldtm_header(file):
  header = """\
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM_LOAD Copy Traits
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)


def emit_ldtm_traits(file, dp, bits, repeat, pack16b, doc=False):
  ldtm_traits = """\
using SM100::TMEM::LOAD::SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b};

template <>
struct Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>
{{
{thr_comment}\
  using ThrID = Layout<{thrid_layout}>;
{val_comment}\
  using ValID = Layout<Shape <{col_shape},       _{dp}>,
                       Stride<{col_stride},TMEM::DP_b>>;
{src_comment}\
  using SrcLayout = Layout<Shape <{src_thr_shape},{src_val_shape}>,
                           Stride<{src_thr_stride},{src_val_stride}>>;
{dst_comment}\
  using DstLayout = Layout<Shape <{dst_thr_shape},{dst_val_shape}>,
                           Stride<{dst_thr_stride},{dst_val_stride}>>;
{ref_comment}\
  using RefLayout = SrcLayout;
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  thr_comment = "  // Logical thread id to thread idx (warp)\n" if doc else ""
  val_comment = "  // Logical bit id to bit idx (address)\n" if doc else ""
  src_comment = "  // Map from (src-thr,src-val) to bit\n" if doc else ""
  dst_comment = "  // Map from (dst-thr,dst-val) to bit\n" if doc else ""
  ref_comment = "  // Reference map from (thr,val) to bit\n" if doc else ""

  is_16dp32b = dp == 16 and bits == 32
  op_16b = "_16b" if pack16b else ""

  dp_bits = (bits * repeat) if not is_16dp32b else (bits * repeat * 2) # bits in one datapath
  instr_bits = dp * dp_bits
  half_bits = instr_bits // 2

  thrid_layout = "_32"

  if pack16b:
    col_shape  = [16, dp_bits // 16]
    col_stride = [1, 32]
  else:
    col_shape  = [dp_bits]
    col_stride = [1]

  src_thr_shape  = [32]
  src_thr_stride = [0]
  src_val_shape  = [instr_bits]
  src_val_stride = [1]

  if bits == 32:
    dst_thr_shape  = [32] if not is_16dp32b else [16, 2]
    dst_thr_stride = [dp_bits] if not is_16dp32b else [dp_bits, dp_bits // 2]
    dst_val_shape  = [dp_bits] if not is_16dp32b else [dp_bits // 2]
    dst_val_stride = [1]
  elif bits == 64:
    dst_thr_shape  = [2, 2, 8]
    dst_thr_stride = [half_bits, 32, dp_bits]
    dst_val_shape  = [32, repeat] if repeat > 1 else [32]
    dst_val_stride = [1, 64] if repeat > 1 else [1]
  elif bits == 128:
    dst_thr_shape  = [4, 8]
    dst_thr_stride = [32, dp_bits]
    dst_val_shape  = [32, 2, repeat] if repeat > 1 else [32, 2]
    dst_val_stride = [1, half_bits, 128] if repeat > 1 else [1, half_bits]
  elif bits == 256:
    dst_thr_shape  = [4, 8]
    dst_thr_stride = [64, dp_bits]
    dst_val_shape  = [64, 2, repeat] if repeat > 1 else [64, 2]
    dst_val_stride = [1, half_bits, 256] if repeat > 1 else [1, half_bits]

  def format_layout(shapes, strides):
    tuple_to_str = lambda t : [('_{}' if e <= 524288 else 'Int<{}>').format(e) for e in t]
    justify = lambda s0, s1 : (s0.rjust(max(len(s0),len(s1))), s1.rjust(max(len(s0),len(s1))))

    shapes, strides = zip(*[justify(*s) for s in zip(tuple_to_str(shapes), tuple_to_str(strides))])
    if len(shapes) == 1:
      return shapes[0], strides[0]
    else:
      return 'Shape <'+','.join(shapes)+'>', 'Stride<'+','.join(strides)+'>'

  col_shape, col_stride = format_layout(col_shape, col_stride)
  src_thr_shape, src_thr_stride = format_layout(src_thr_shape, src_thr_stride)
  src_val_shape, src_val_stride = format_layout(src_val_shape, src_val_stride)
  dst_thr_shape, dst_thr_stride = format_layout(dst_thr_shape, dst_thr_stride)
  dst_val_shape, dst_val_stride = format_layout(dst_val_shape, dst_val_stride)

  file.write(ldtm_traits.format(**locals()))

def emit_sttm_header(file):
  header = """\
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM_STORE Copy Traits
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)

def emit_sttm_traits(file, dp, bits, repeat, expand16b):
  sttm_traits = """\
using SM100::TMEM::STORE::SM100_TMEM_STORE_{dp}dp{bits}b{repeat}x{op_16b};

template <>
struct Copy_Traits<SM100_TMEM_STORE_{dp}dp{bits}b{repeat}x{op_16b}>
{{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{op_16b}>::RefLayout;
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  op_16b = "_16b" if expand16b else ""

  file.write(sttm_traits.format(**locals()))


def generate_sm100_copy_traits(args: argparse.Namespace):
  # The repeating variants available
  repeats = [1, 2, 4, 8, 16, 32, 64, 128]

  # Emit to file
  emit_section_header(args.outfile)

  # Emit TMEM_LOAD Traits
  emit_ldtm_header(args.outfile)

  # TMEM_LOAD_16dp256b
  for repeat in repeats[:-2]:
    emit_ldtm_traits(args.outfile, 16, 256, repeat, pack16b=False, doc=(repeat == 1))
    emit_ldtm_traits(args.outfile, 16, 256, repeat, pack16b=True)

  # TMEM_LOAD_16dp128b
  for repeat in repeats[:-1]:
    emit_ldtm_traits(args.outfile, 16, 128, repeat, pack16b=False)
    emit_ldtm_traits(args.outfile, 16, 128, repeat, pack16b=True)

  # TMEM_LOAD_16dp64b
  for repeat in repeats:
    emit_ldtm_traits(args.outfile, 16, 64, repeat, pack16b=False)
    emit_ldtm_traits(args.outfile, 16, 64, repeat, pack16b=True)

  # TMEM_LOAD_16dp32b
  for repeat in repeats:
    emit_ldtm_traits(args.outfile, 16, 32, repeat, pack16b=False)
    emit_ldtm_traits(args.outfile, 16, 32, repeat, pack16b=True)

  # TMEM_LOAD_32dp32b
  for repeat in repeats:
    emit_ldtm_traits(args.outfile, 32, 32, repeat, pack16b=False)
    emit_ldtm_traits(args.outfile, 32, 32, repeat, pack16b=True)


  # Emit TMEM_STORE Traits
  emit_sttm_header(args.outfile)

  # TMEM_STORE_16dp256b
  for repeat in repeats[:-2]:
    emit_sttm_traits(args.outfile, 16, 256, repeat, expand16b=False)
    emit_sttm_traits(args.outfile, 16, 256, repeat, expand16b=True)

  # TMEM_STORE_16dp128b
  for repeat in repeats[:-1]:
    emit_sttm_traits(args.outfile, 16, 128, repeat, expand16b=False)
    emit_sttm_traits(args.outfile, 16, 128, repeat, expand16b=True)

  # TMEM_STORE_16dp64b
  for repeat in repeats:
    emit_sttm_traits(args.outfile, 16, 64, repeat, expand16b=False)
    emit_sttm_traits(args.outfile, 16, 64, repeat, expand16b=True)

  # TMEM_STORE_16dp32b
  for repeat in repeats:
    emit_sttm_traits(args.outfile, 16, 32, repeat, expand16b=False)
    emit_sttm_traits(args.outfile, 16, 32, repeat, expand16b=True)

  # TMEM_STORE_32dp32b
  for repeat in repeats:
    emit_sttm_traits(args.outfile, 32, 32, repeat, expand16b=False)
    emit_sttm_traits(args.outfile, 32, 32, repeat, expand16b=True)

  # Emit op selector
  emit_namespace_header(args.outfile, 'TMEM')
  emit_op_selector_header(args.outfile)

  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 256, 256, repeats[:-2], pack16b=False, else_='')
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 256, 256, repeats[:-2], pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 128, 128, repeats[:-1], pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 128, 128, repeats[:-1], pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 64, 64, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 64, 64, repeats, pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 32, 64, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 16, 32, 64, repeats, pack16b=True)
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 32, 32, 32, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_LOAD', 32, 32, 32, repeats, pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 256, 256, repeats[:-2], pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 256, 256, repeats[:-2], pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 128, 128, repeats[:-1], pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 128, 128, repeats[:-1], pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 64, 64, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 64, 64, repeats, pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 32, 64, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_STORE', 16, 32, 64, repeats, pack16b=True)

  emit_op_selector_body(args.outfile, 'TMEM_STORE', 32, 32, 32, repeats, pack16b=False)
  emit_op_selector_body(args.outfile, 'TMEM_STORE', 32, 32, 32, repeats, pack16b=True)

  emit_op_selector_footer(args.outfile)
  emit_namespace_footer(args.outfile, 'TMEM')

  emit_section_footer(args.outfile)

  if args.outfile is not sys.stdout:
    args.outfile.close()

  print("// Generated TMEM_LOAD/TMEM_STORE Copy Traits.")
  return


def parse_cmd(argv: List[str]):
  parser = argparse.ArgumentParser(
      "Argument parser for SM100 Copy Traits Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which copy traits will be written.")
  return parser.parse_args(argv)


def main():
  args = parse_cmd(sys.argv[1:])
  generate_sm100_copy_traits(args)


if __name__ == "__main__":
  main()
