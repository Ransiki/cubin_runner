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
from itertools import product
import sys
from typing import *

from generator_library import *
from mma_atom_generator import emit_mma_atom_wrapper


CUTE_MMA_ATOM_HEADER_TEMPLATE = "// {mma_tag} {m}x{n}x{k} {transA}{transB} {atype} x {btype}"
CUTE_MMA_ATOM_NAME_TEMPLATE = "SM{sm_arch}_{m}x{n}x{k}_{transA}{transB}"
CUTE_MMA_ATOM_WRAPPER_TEMPLATE = """\
template <>
struct {mma_atom_name}<{a_type}, {b_type}, {c_type}>
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


CUTE_BLOCKSCALED_MMA_ATOM_HEADER_TEMPLATE = "// {mma_tag} {m}x{n}x{k} {transA}{transB} {atype} x {btype} with SF {sftype}"
CUTE_BLOCKSCALED_MMA_ATOM_NAME_TEMPLATE = "SM{sm_arch}_{m}x{n}x{k}_{transA}{transB}_VS"
CUTE_BLOCKSCALED_QMMA_ATOM_WRAPPER_TEMPLATE = """\
template <int VS>
struct {mma_atom_name}<{a_type}, {b_type}, {c_type}, {sf_type}, VS>
{{
  using DRegisters = {d_regs};
  using ARegisters = {a_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};
  using SFARegisters = {sfa_regs};
  using SFBRegisters = {sfb_regs};

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for QMMA.");

    asm volatile(
    {ptx_instruction_vs32}
    {d_ptx_operands}
    {a_ptx_operands}
    {b_ptx_operands}
    {c_ptx_operands}
    {sfa_ptx_operands}
    {sfa_index_ptx_operands}
    {sfb_ptx_operands}
    {sfb_index_ptx_operands}
    {ptx_bindings});

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::{mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}_ENABLED");
#endif
  }}
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""

CUTE_BLOCKSCALED_OMMA_ATOM_WRAPPER_TEMPLATE = """\
template <int VS>
struct {mma_atom_name}<{a_type}, {b_type}, {c_type}, {sf_type}, VS>
{{
  using DRegisters = {d_regs};
  using ARegisters = {a_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};
  using SFARegisters = {sfa_regs};
  using SFBRegisters = {sfb_regs};

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

 CUTE_STATIC_ASSERT(VS == 16 || VS == 32, "Scaling factor vector size has to be 16 or 32 for OMMA.");
    if constexpr ( VS == 16 ) {{
      asm volatile(
      {ptx_instruction_vs16}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {c_ptx_operands}
      {sfa_ptx_operands}
      {sfa_index_ptx_operands}
      {sfb_ptx_operands}
      {sfb_index_ptx_operands}
      {ptx_bindings});
    }} else if constexpr ( VS == 32 ) {{
      asm volatile(
      {ptx_instruction_vs32}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {c_ptx_operands}
      {sfa_ptx_operands}
      {sfa_index_ptx_operands}
      {sfb_ptx_operands}
      {sfb_index_ptx_operands}
      {ptx_bindings});
    }}

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::{mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}_ENABLED");
#endif
  }}
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
def emit_cute_file_header(file):
  emit_license_header(file)
  header = """\

//
// {$nv-internal-release file}
//

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200))
#  define CUTE_ARCH_MMA_SM120_ENABLED
#endif

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(header)


def mma_atom_template(atom: MmaAtomConfig) -> str:
  trans_str = lambda trans_bool : 'T' if trans_bool else 'N'
  return CUTE_MMA_ATOM_NAME_TEMPLATE.format(
    sm_arch=120, m=atom.m, n=atom.n, k=atom.k,
    transA=trans_str(atom.transA), transB=trans_str(atom.transB))


def emit_default_template(file, atom_template: str):
  template_str = f"""\
template <class a_type, class b_type, class c_type>
struct {atom_template}
{{
  static_assert(sizeof(a_type) == 0, "No MMA matches {atom_template} for given data types.");
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(template_str)

def blockscaled_mma_atom_template(atom: MmaAtomConfig) -> str:
  trans_str = lambda trans_bool : 'T' if trans_bool else 'N'
  return CUTE_BLOCKSCALED_MMA_ATOM_NAME_TEMPLATE.format(
    sm_arch=120, m=atom.m, n=atom.n, k=atom.k,
    transA=trans_str(atom.transA), transB=trans_str(atom.transB))

def emit_default_blockscaled_template(file, atom_template: str):
  template_str = f"""\
template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct {atom_template}
{{
  static_assert(sizeof(a_type) == 0, "No MMA matches {atom_template} for given data types.");
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

"""
  file.write(template_str)


def emit_cute_file_footer(file):
  footer = """\
} // namespace cute
"""
  file.write(footer)


def generate_sm120_mma_atoms(args: argparse.Namespace):
  atoms = []

  arch = 120

  fourbit_ab_types = [DataType.e2m1, DataType.e0m3]
  qmma_qualified_fourbit_ab_types = [DataType.e2m1] # Only e2m1 is used in QMMA
  sixbit_ab_types = [DataType.e3m2, DataType.e2m3]
  eightbit_ab_types = [DataType.e4m3, DataType.e5m2, DataType.e3m4]

  #
  # Generate QMMAs
  #

  eightbit_mk_sizes = [[16, 32]]
  modifiers = [MmaPtxModifier.kNone]
  qmma_types = qmma_qualified_fourbit_ab_types + sixbit_ab_types + eightbit_ab_types
  combs = product(qmma_types, qmma_types, eightbit_mk_sizes, modifiers)
  for a_type, b_type, mk, mod in combs:
    atoms.append(MmaAtomConfig(
      "MMA", mk[0], 8, mk[1], a_type, MmaOperandSource.reg, b_type, MmaOperandSource.reg,
      DataType.f32, MmaOperandSource.reg, 32, True, False, 0, arch, mod))

  # Emit to file
  emit_cute_file_header(args.outfile)
  emitted_templates = set()
  for atom in atoms:
    # Emit top-level template, if needed
    atom_template = mma_atom_template(atom)
    if atom_template not in emitted_templates:
      emit_default_template(args.outfile, atom_template)
      emitted_templates.add(atom_template)

    is_f8f6f4 = atom.m == 16 and atom.n == 8 and atom.k == 32
    emit_mma_atom_wrapper(
      args.outfile,
      CUTE_MMA_ATOM_HEADER_TEMPLATE,
      CUTE_MMA_ATOM_NAME_TEMPLATE,
      CUTE_MMA_ATOM_WRAPPER_TEMPLATE,
      atom,
      fma_args_indent=6,
      ptx_indent=6,
      round_up_ab_sizes=is_f8f6f4)

  emit_cute_file_footer(args.outfile)

  args.outfile.close()


def generate_sm120_blockscaled_mma_atoms(args: argparse.Namespace):
  atoms = []

  arch = 120

  fourbit_ab_types = [DataType.e2m1, DataType.e0m3]
  qmma_qualified_fourbit_ab_types = [DataType.e2m1] # Only e2m1 is used in QMMA
  sixbit_ab_types = [DataType.e3m2, DataType.e2m3]
  eightbit_ab_types = [DataType.e4m3, DataType.e5m2, DataType.e3m4]
  sf_types = [DataType.ue8m0]
 
  #
  # Generate QMMAs
  #

  eightbit_mk_sizes = [[16, 32]]
  modifiers = [MmaPtxModifier.kNone]
  qmma_types = qmma_qualified_fourbit_ab_types + sixbit_ab_types + eightbit_ab_types
  vec_sizes = [32]
  combs = product(qmma_types, qmma_types, eightbit_mk_sizes, modifiers, sf_types)
  for a_type, b_type, mk, mod, sf_type in combs:
    atoms.append(MmaAtomConfig(
      "QMMA.SF", mk[0], 8, mk[1], a_type, MmaOperandSource.reg, b_type, MmaOperandSource.reg,
      DataType.f32, MmaOperandSource.reg, 32, True, False, 0, arch, mod, vec_sizes, sf_type, MmaOperandSource.reg))


  # Emit to file
  emit_cute_file_header(args.outfile)
  emitted_templates = set()
  for atom in atoms:
    # Emit top-level template, if needed
    atom_template = blockscaled_mma_atom_template(atom)
    if atom_template not in emitted_templates:
      emit_default_blockscaled_template(args.outfile, atom_template)
      emitted_templates.add(atom_template)

    is_f8f6f4 = atom.m == 16 and atom.n == 8 and atom.k == 32
    emit_mma_atom_wrapper(
      args.outfile,
      CUTE_BLOCKSCALED_MMA_ATOM_HEADER_TEMPLATE,
      CUTE_BLOCKSCALED_MMA_ATOM_NAME_TEMPLATE,
      CUTE_BLOCKSCALED_QMMA_ATOM_WRAPPER_TEMPLATE,
      atom,
      fma_args_indent=6,
      ptx_indent=4,
      round_up_ab_sizes=is_f8f6f4)


  #
  # Generate QMMAs
  #

  eightbit_mk_sizes = [[16, 64]]
  modifiers = [MmaPtxModifier.kNone]
  omma_types = fourbit_ab_types
  vec_sizes = [16, 32]
  combs = product(omma_types, omma_types, eightbit_mk_sizes, modifiers, sf_types)
  atoms = []
  for a_type, b_type, mk, mod, sf_type in combs:
    atoms.append(MmaAtomConfig(
      "OMMA.SF", mk[0], 8, mk[1], a_type, MmaOperandSource.reg, b_type, MmaOperandSource.reg,
      DataType.f32, MmaOperandSource.reg, 32, True, False, 0, arch, mod, vec_sizes, sf_type, MmaOperandSource.reg))
  
  for atom in atoms:
    # Emit top-level template, if needed
    atom_template = blockscaled_mma_atom_template(atom)
    if atom_template not in emitted_templates:
      emit_default_blockscaled_template(args.outfile, atom_template)
      emitted_templates.add(atom_template)

    emit_mma_atom_wrapper(
      args.outfile,
      CUTE_BLOCKSCALED_MMA_ATOM_HEADER_TEMPLATE,
      CUTE_BLOCKSCALED_MMA_ATOM_NAME_TEMPLATE,
      CUTE_BLOCKSCALED_OMMA_ATOM_WRAPPER_TEMPLATE,
      atom,
      fma_args_indent=6,
      ptx_indent=6,
      round_up_ab_sizes=False)


  emit_cute_file_footer(args.outfile)

  args.outfile.close()

def parse_cmd():
  parser = argparse.ArgumentParser(
      "Argument parser for SM120 MMA Atom Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which MMA atoms will be written.")
  return parser.parse_args()


def main():
  args = parse_cmd()
  generate_sm120_mma_atoms(args)
  generate_sm120_blockscaled_mma_atoms(args)

if __name__ == "__main__":
  main()