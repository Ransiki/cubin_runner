#!/usr/bin/python3
#################################################################################################
#
# Copyright (c) 2020 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os.path as osp
from typing import *
from generator_library import *

# Atom generator
from mma_atom_generator import emit_mma_atom_wrapper

CUTE_MMA_ATOM_HEADER_TEMPLATE = "// {mma_tag} {m}x{n}x{k} {transA}{transB}"
CUTE_MMA_ATOM_NAME_TEMPLATE = "SM{sm_arch}_{m}x{n}x{k}_{cdtype}{atype}{btype}{cdtype}_{transA}{transB}"
CUTE_MMA_ATOM_WRAPPER_TEMPLATE = """\
struct {mma_atom_name}
{{
  using DRegisters = {d_regs};
  using ARegisters = {a_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}_ENABLED)
    asm volatile(
      {ptx_instruction}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {c_ptx_operands}
      {ptx_bindings});
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use {mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}_ENABLED");
#endif
  }}
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""


def emit_cute_file_header(file):
  emit_license_header(file)
  header = """\

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#  define CUTE_ARCH_MMA_SM80_ENABLED
#endif

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)


def emit_cute_file_footer(file):
  footer = """\
} // namespace cute
"""
  file.write(footer)


def emit_cute_sm80_atom(outfile, atom_config: MmaAtomConfig):
  emit_mma_atom_wrapper(
      outfile,
      CUTE_MMA_ATOM_HEADER_TEMPLATE,
      CUTE_MMA_ATOM_NAME_TEMPLATE,
      CUTE_MMA_ATOM_WRAPPER_TEMPLATE,
      atom_config,
      fma_args_indent=6,
      ptx_indent=6)


def generate_sm80_mma_atoms(args: argparse.Namespace):
  atoms = []

  # f16 -> f16
  atoms.extend([MmaAtomConfig(
      "MMA", 16, 8, k, DataType.f16, MmaOperandSource.reg, DataType.f16,
      MmaOperandSource.reg, DataType.f16, MmaOperandSource.reg, 32, True, False, 0, 80)
      for k in [8, 16]])

  # f16|bfloat -> f32
  for src_type in [DataType.f16, DataType.bf16]:
    for k in [8, 16]:
      atoms.append(MmaAtomConfig(
        "MMA", 16, 8, k, src_type, MmaOperandSource.reg, src_type,
        MmaOperandSource.reg, DataType.f32, MmaOperandSource.reg, 32, True, False, 0, 80))

  # tf32 -> f32
  atoms.extend([
      MmaAtomConfig(
          "MMA", 16, 8, k, DataType.tf32, MmaOperandSource.reg, DataType.tf32,
          MmaOperandSource.reg, DataType.f32, MmaOperandSource.reg, 32, True, False, 0, 80)
      for k in [4, 8]])

  # f64
  atoms.append(MmaAtomConfig(
      "MMA", 8, 8, 4, DataType.f64, MmaOperandSource.reg, DataType.f64,
      MmaOperandSource.reg, DataType.f64, MmaOperandSource.reg, 32, True, False, 0, 80))

  # int
  int8_ab_types = [DataType.s8, DataType.u8]
  int8_mk_sizes = [[8, 16], [16, 16], [16, 32]]
  modifiers = [MmaPtxModifier.kNone, MmaPtxModifier.kSaturate]
  for a_type in int8_ab_types:
    for b_type in int8_ab_types:
      for mk in int8_mk_sizes:
        for mod in modifiers:
          atoms.append(MmaAtomConfig(
            "MMA", mk[0], 8, mk[1], a_type, MmaOperandSource.reg, b_type, MmaOperandSource.reg,
            DataType.s32, MmaOperandSource.reg, 32, True, False, 0, 80, mod))

  int4_ab_types = [DataType.s4, DataType.u4]
  int4_mk_sizes = [[mk[0], 2*mk[1]] for mk in int8_mk_sizes]
  for a_type in int4_ab_types:
    for b_type in int4_ab_types:
      for mk in int4_mk_sizes:
        for mod in modifiers:
          atoms.append(MmaAtomConfig(
            "MMA", mk[0], 8, mk[1], a_type, MmaOperandSource.reg, b_type, MmaOperandSource.reg,
            DataType.s32, MmaOperandSource.reg, 32, True, False, 0, 80, mod))

  bmma_types = [MmaPtxModifier.kAndPopc, MmaPtxModifier.kXorPopc]
  bmma_sizes = [[8, 8, 128], [16, 8, 128], [16, 8, 256]]
  for bmma_type in bmma_types:
    for mnk in bmma_sizes:
      atoms.append(MmaAtomConfig(
        "MMA", mnk[0], mnk[1], mnk[2],
        DataType.b1,  MmaOperandSource.reg,
        DataType.b1,  MmaOperandSource.reg,
        DataType.s32, MmaOperandSource.reg,
        32, True, False, 0, 80, bmma_type
      ))

  # Emit to file
  emit_cute_file_header(args.outfile)
  for atom in atoms:
    emit_cute_sm80_atom(args.outfile, atom)
  emit_cute_file_footer(args.outfile)

  args.outfile.close()
  return


def parse_cmd(argv: List[str]):
  parser = argparse.ArgumentParser(
      "Argument parser for SM80 MMA Atom Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which MMA atoms will be written.")
  return parser.parse_args(argv)


def main():
  args = parse_cmd(sys.argv[1:])
  generate_sm80_mma_atoms(args)


if __name__ == "__main__":
  main()
