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

def emit_ldtm_header(file):
  header = """\
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM LOAD PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)


def emit_ldtm_op(file, dp, bits, repeat, pack16b):
  ldtm_op = """\
// {dp} data path lanes, {bits}-bit pattern, repeated {repeat} times{pack16b_doc}
struct SM100_TMEM_LOAD_{dp}dp{bits}b{repeat}x{pack16b_op}
{{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[{regs}];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       {args})
  {{
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.{dp}x{bits}b{threads_ptx}.x{repeat}{pack16b_ptx}.b32"
                    {dst_regs}
                    "[{src_addr}]{thread_offset};\\n"
    :  {dst_vars}
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }}
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  regs = ((bits+63) // 64) * repeat
  digits = len(str(regs))

  args = ""
  dst_regs = "\"{"
  dst_vars = ""
  for i in range(regs):
    if i > 0:
      args += "\n       " if (i % 4 == 0) else " "
      dst_regs += "\"\n                    \"" if (i % 4 == 0) else " "
      dst_vars += "\n       " if (i % 4 == 0) else " "
    args += "uint32_t& dst{i:0{digits}d}".format(i=i,digits=digits)
    dst_regs += "%" + str(i)
    dst_vars += "\"=r\"(dst{i:0{digits}d})".format(i=i,digits=digits)
    if (i != regs-1):
      args += ","
      dst_regs += ","
      dst_vars += ","
    if (i == regs-1):
      dst_regs += "},\""

  src_addr = "%" + str(regs)

  is_16dp32b = dp == 16 and bits == 32
  threads_ptx = "x2" if is_16dp32b else ""
  thread_offset = ", " + (str(repeat) if not pack16b else str(repeat*2)) if is_16dp32b else ""

  pack16b_doc = ", packed 16b read" if pack16b else ""
  pack16b_op = "_16b" if pack16b else ""
  pack16b_ptx = ".pack::16b" if pack16b else ""

  file.write(ldtm_op.format(
              dp=dp,
              bits=bits,
              repeat=repeat,
              threads_ptx=threads_ptx,
              pack16b_doc=pack16b_doc,
              pack16b_op=pack16b_op,
              pack16b_ptx=pack16b_ptx,
              regs=regs,
              args=args,
              dst_regs=dst_regs,
              src_addr=src_addr,
              dst_vars=dst_vars,
              thread_offset=thread_offset))

def emit_sttm_header(file):
  header = """\
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM STORE PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)

def emit_sttm_op(file, dp, bits, repeat, expand16b):
  sttm_op = """\
// {dp} data path lanes, {bits}-bit pattern, repeated {repeat} times{expand16b_doc}
struct SM100_TMEM_STORE_{dp}dp{bits}b{repeat}x{expand16b_op}
{{
  using SRegisters = uint32_t[{regs}];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy({args},
       uint32_t const& dst_addr)
  {{
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.{dp}x{bits}b{threads_ptx}.x{repeat}{expand16b_ptx}.b32"
                    "[%0]{thread_offset},"
                    {src_regs}
    :
    :  "r"(dst_addr), {src_vars} );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }}
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  regs = ((bits+63) // 64) * repeat
  digits = len(str(regs))

  args = ""
  src_regs = "\"{"
  src_vars = ""
  for i in range(regs):
    if i > 0:
      args += "\n       " if (i % 4 == 0) else " "
      src_regs += "\"\n                    \"" if (i % 4 == 0) else " "
      src_vars += "\n       " if (i % 4 == 0) else " "
    args += "uint32_t const& src{i:0{digits}d}".format(i=i,digits=digits)
    src_regs += "%" + str(i+1)
    src_vars += "\"r\"(src{i:0{digits}d})".format(i=i,digits=digits)
    if (i != regs-1):
      args += ","
      src_regs += ","
      src_vars += ","
    if (i == regs-1):
      src_regs += "};\\n\""

  dst_addr = "%" + str(0)

  is_16dp32b = dp == 16 and bits == 32
  threads_ptx = "x2" if is_16dp32b else ""
  thread_offset = " , " + (str(repeat) if not expand16b else str(repeat*2)) if is_16dp32b else ""

  expand16b_doc = ", expand 16b write" if expand16b else ""
  expand16b_op = "_16b" if expand16b else ""
  expand16b_ptx = ".unpack::16b" if expand16b else ""

  file.write(sttm_op.format(
              dp=dp,
              bits=bits,
              repeat=repeat,
              threads_ptx=threads_ptx,
              expand16b_doc=expand16b_doc,
              expand16b_op=expand16b_op,
              expand16b_ptx=expand16b_ptx,
              regs=regs,
              args=args,
              src_regs=src_regs,
              dst_addr=dst_addr,
              src_vars=src_vars,
              thread_offset=thread_offset))

def generate_sm100_copy_ops(args: argparse.Namespace):
  # The repeating variants available
  repeats = [1, 2, 4, 8, 16, 32, 64, 128]

  # Emit to file
  emit_section_header(args.outfile)

  # Emit TMEM_LOAD Operators
  emit_namespace_header(args.outfile, 'SM100::TMEM::LOAD')
  emit_ldtm_header(args.outfile)

  # TMEM_LOAD_16dp256b
  for repeat in repeats[:-2]:
    emit_ldtm_op(args.outfile, 16, 256, repeat, pack16b=False)
    emit_ldtm_op(args.outfile, 16, 256, repeat, pack16b=True)

  # TMEM_LOAD_16dp128b
  for repeat in repeats[:-1]:
    emit_ldtm_op(args.outfile, 16, 128, repeat, pack16b=False)
    emit_ldtm_op(args.outfile, 16, 128, repeat, pack16b=True)

  # TMEM_LOAD_16dp64b
  for repeat in repeats:
    emit_ldtm_op(args.outfile, 16, 64, repeat, pack16b=False)
    emit_ldtm_op(args.outfile, 16, 64, repeat, pack16b=True)

  # TMEM_LOAD_16dp32b
  for repeat in repeats:
    emit_ldtm_op(args.outfile, 16, 32, repeat, pack16b=False)
    emit_ldtm_op(args.outfile, 16, 32, repeat, pack16b=True)

  # TMEM_LOAD_32dp32b
  for repeat in repeats:
    emit_ldtm_op(args.outfile, 32, 32, repeat, pack16b=False)
    emit_ldtm_op(args.outfile, 32, 32, repeat, pack16b=True)

  emit_namespace_footer(args.outfile, 'SM100::TMEM::LOAD')

  # Emit TMEM_STORE Operators
  emit_namespace_header(args.outfile, 'SM100::TMEM::STORE')
  emit_sttm_header(args.outfile)

  # TMEM_STORE_16dp256b
  for repeat in repeats[:-2]:
    emit_sttm_op(args.outfile, 16, 256, repeat, expand16b=False)
    emit_sttm_op(args.outfile, 16, 256, repeat, expand16b=True)

  # TMEM_STORE_16dp128b
  for repeat in repeats[:-1]:
    emit_sttm_op(args.outfile, 16, 128, repeat, expand16b=False)
    emit_sttm_op(args.outfile, 16, 128, repeat, expand16b=True)

  # TMEM_STORE_16dp64b
  for repeat in repeats:
    emit_sttm_op(args.outfile, 16, 64, repeat, expand16b=False)
    emit_sttm_op(args.outfile, 16, 64, repeat, expand16b=True)

  # TMEM_STORE_16dp32b
  for repeat in repeats:
    emit_sttm_op(args.outfile, 16, 32, repeat, expand16b=False)
    emit_sttm_op(args.outfile, 16, 32, repeat, expand16b=True)

  # TMEM_STORE_32dp32b
  for repeat in repeats:
    emit_sttm_op(args.outfile, 32, 32, repeat, expand16b=False)
    emit_sttm_op(args.outfile, 32, 32, repeat, expand16b=True)

  emit_namespace_footer(args.outfile, 'SM100::TMEM::STORE')
  emit_section_footer(args.outfile)

  if args.outfile is not sys.stdout:
    args.outfile.close()

  print("// Generated TMEM_LOAD/TMEM_STORE Copy Operators.")
  return


def parse_cmd(argv: List[str]):
  parser = argparse.ArgumentParser(
      "Argument parser for SM100 Copy Operator Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which copy ops will be written.")
  return parser.parse_args(argv)


def main():
  args = parse_cmd(sys.argv[1:])
  generate_sm100_copy_ops(args)


if __name__ == "__main__":
  main()
