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
from mma_atom_generator import emit_mma_atom_wrapper, sm90_gmma_n_sizes

CUTE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE = """\
struct {mma_atom_name}
{{
  using DRegisters = void;
  using ARegisters = {a_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_{synclog_a_tag}_{synclog_b_tag}(__LINE__{synclog_a_arg}{synclog_b_arg});
    asm volatile(
    "{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, {scaleD_ptx_operand}, 0;\\n"
      {ptx_instruction}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {mod_ptx_operands}
    "}}\\n"
      {ptx_bindings});
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use {mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED");
#endif
  }}
}};
"""

CUTE_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE = """\
struct {mma_atom_name}
{{
  using DRegisters = void;
  using ARegisters = {a_regs};
  using ERegisters = {e_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_{synclog_a_tag}_{synclog_b_tag}(__LINE__{synclog_a_arg}{synclog_b_arg});
    asm volatile(
    "{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, {scaleD_ptx_operand}, 0;\\n"
      {ptx_instruction}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {e_ptx_operands}
      {mod_ptx_operands}
    "}}\\n"
      {ptx_bindings});
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM90::GMMA::SPARSE::{mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED");
#endif
  }}
}};
"""

CUTE_RS_GMMA_ATOM_WRAPPER_TEMPLATE_BASE = """\
struct {mma_atom_name}
{{
  using DRegisters = void;
  using ARegisters = {a_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};

  static_assert(tnspA == GMMA::Major::K,
      "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_{synclog_a_tag}_{synclog_b_tag}(__LINE__{synclog_a_arg}{synclog_b_arg});
    asm volatile(
    "{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, {scaleD_ptx_operand}, 0;\\n"
      {ptx_instruction}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {mod_ptx_operands}
    "}}\\n"
      {ptx_bindings});
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use {mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED");
#endif
  }}
}};
"""

CUTE_RS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE = """\
struct {mma_atom_name}
{{
  using DRegisters = void;
  using ARegisters = {a_regs};
  using ERegisters = {e_regs};
  using BRegisters = {b_regs};
  using CRegisters = {c_regs};

  static_assert(tnspA == GMMA::Major::K,
      "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma({fma_exploded_args})
  {{
#if defined(CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_{synclog_a_tag}_{synclog_b_tag}(__LINE__{synclog_a_arg}{synclog_b_arg});
    asm volatile(
    "{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, {scaleD_ptx_operand}, 0;\\n"
      {ptx_instruction}
      {d_ptx_operands}
      {a_ptx_operands}
      {b_ptx_operands}
      {e_ptx_operands}
      {mod_ptx_operands}
    "}}\\n"
      {ptx_bindings});
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM90::GMMA::SPARSE::{mma_atom_name} without CUTE_ARCH_MMA_SM{sm_arch}A_ENABLED");
#endif
  }}
}};
"""

CUTE_Int8_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join([
CUTE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_Int8_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>""",
CUTE_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_Fp8_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>""",
CUTE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_Fp8_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One,
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>""",
CUTE_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_16b_SS_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>""",
CUTE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_16b_SS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One,
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>""",
CUTE_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_16b_RS_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>""",
CUTE_RS_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_16b_RS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One,
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>""",
CUTE_RS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_32b_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>""",
CUTE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_32b_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE = '\n'.join(["""\
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One,
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>""",
CUTE_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE_BASE])

CUTE_hGMMA_ATOM_HEADER_TEMPLATE = "// {mma_tag} {m}x{n}x{k} {cdtype}+={atype}*{btype}"
CUTE_xMMA_ATOM_HEADER_TEMPLATE = "// {mma_tag} {m}x{n}x{k} {transA}{transB} {cdtype}+={atype}*{btype}"
CUTE_hGMMA_ATOM_NAME_TEMPLATE = "MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"
CUTE_xGMMA_ATOM_NAME_TEMPLATE = "MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_hGMMA_ATOM_NAME_TEMPLATE_SPARSE = "GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"
CUTE_xGMMA_ATOM_NAME_TEMPLATE_SPARSE = "GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"


def emit_cute_file_header(file):
  emit_license_header(file)
  header = """\
  
#include <cute/config.hpp>                // CUTE_HOST_DEVICE

#include "cutlass/arch/synclog.hpp"

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#  define CUTE_ARCH_MMA_SM90A_ENABLED
#endif

namespace cute {

"""
  file.write(header)


def emit_cute_file_footer(file):
  footer = """\
} // namespace cute
"""
  file.write(footer)


@post_separator
def emit_cute_sm90_32b_gmma_atom(outfile, atom_config: MmaAtomConfig):
  if atom_config.kind == MmaKind.sparse:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE_SPARSE
    mma_atom_wrapper_template = CUTE_32b_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE
  else:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE
    mma_atom_wrapper_template = CUTE_32b_GMMA_ATOM_WRAPPER_TEMPLATE
  emit_mma_atom_wrapper(
      outfile,
      CUTE_xMMA_ATOM_HEADER_TEMPLATE,
      mma_atom_name_template,
      mma_atom_wrapper_template,
      atom_config,
      fma_args_indent=6,
      ptx_indent=6)


@post_separator
def emit_cute_sm90_16b_gmma_atom(outfile, atom_config: MmaAtomConfig):
  if atom_config.kind == MmaKind.sparse:
    mma_atom_name_template    = CUTE_hGMMA_ATOM_NAME_TEMPLATE_SPARSE
    mma_atom_wrapper_template = (CUTE_16b_SS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE 
                                 if atom_config.asource == MmaOperandSource.smem_desc 
                                 else CUTE_16b_RS_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE)
  else:
    mma_atom_name_template    = CUTE_hGMMA_ATOM_NAME_TEMPLATE
    mma_atom_wrapper_template = (CUTE_16b_SS_GMMA_ATOM_WRAPPER_TEMPLATE 
                                 if atom_config.asource == MmaOperandSource.smem_desc 
                                 else CUTE_16b_RS_GMMA_ATOM_WRAPPER_TEMPLATE)
  emit_mma_atom_wrapper(
    outfile,
    CUTE_hGMMA_ATOM_HEADER_TEMPLATE,
    mma_atom_name_template,
    mma_atom_wrapper_template,
    atom_config,
    fma_args_indent=6,
    ptx_indent=6)


@post_separator
def emit_cute_sm90_int8_gmma_atom(outfile, atom_config: MmaAtomConfig):
  if atom_config.kind == MmaKind.sparse:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE_SPARSE
    mma_atom_wrapper_template = CUTE_Int8_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE
  else:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE
    mma_atom_wrapper_template = CUTE_Int8_GMMA_ATOM_WRAPPER_TEMPLATE
  emit_mma_atom_wrapper(
      outfile,
      CUTE_xMMA_ATOM_HEADER_TEMPLATE,
      mma_atom_name_template,
      mma_atom_wrapper_template,
      atom_config,
      fma_args_indent=6,
      ptx_indent=6)


@post_separator
def emit_cute_sm90_fp8_gmma_atom(outfile, atom_config: MmaAtomConfig):
  if atom_config.kind == MmaKind.sparse:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE_SPARSE
    mma_atom_wrapper_template = CUTE_Fp8_SPARSE_GMMA_ATOM_WRAPPER_TEMPLATE
  else:
    mma_atom_name_template    = CUTE_xGMMA_ATOM_NAME_TEMPLATE
    mma_atom_wrapper_template = CUTE_Fp8_GMMA_ATOM_WRAPPER_TEMPLATE
  emit_mma_atom_wrapper(
      outfile,
      CUTE_xMMA_ATOM_HEADER_TEMPLATE,
      mma_atom_name_template,
      mma_atom_wrapper_template,
      atom_config,
      fma_args_indent=6,
      ptx_indent=6)


def generate_sm90_mma_atoms(args: argparse.Namespace):
  gmma_a_source_types = [MmaOperandSource.smem_desc, MmaOperandSource.reg]
  bsource = MmaOperandSource.smem_desc

  # hGMMA: {fp32|fp16} <- {fp16|bf16}
  gmma_atoms_16b = []
  hgmma_ab_types = [DataType.f16, DataType.bf16]
  hgmma_cd_types = [DataType.f16, DataType.f32]
  for ab_type in hgmma_ab_types:
    for cd_type in hgmma_cd_types:
      # make sure fp16 dest only has fp16 source
      if cd_type == DataType.f16 and ab_type != DataType.f16:
        continue
      for n in sm90_gmma_n_sizes(ab_type, args.extended):
        for asource in gmma_a_source_types:
          if not args.sparse:
            gmma_atoms_16b.append(MmaAtomConfig(
                "GMMA",
                64, n, 16,
                ab_type, asource, ab_type, bsource, cd_type, MmaOperandSource.reg,
                128, None, None, 0, 90))
          else:
            gmma_atoms_16b.append(MmaAtomConfig(
                "SPARSE GMMA",
                64, n, 32,
                ab_type, asource, ab_type, bsource, cd_type, MmaOperandSource.reg,
                128, None, None, 0, 90,
                kind=MmaKind.sparse))

  # hGMMA - fp32 <- tf32
  # tf32 only supports a single set of dstfmt, transA, transB
  gmma_atoms_32b = []
  for n in sm90_gmma_n_sizes(DataType.tf32, args.extended):
    for asource in gmma_a_source_types:
      if not args.sparse:
        gmma_atoms_32b.append(MmaAtomConfig(
            "GMMA", 64, n, 8,
            DataType.tf32, asource,
            DataType.tf32, bsource,
            DataType.f32, MmaOperandSource.reg,
            128, True, False, 0, 90))
      else:
        gmma_atoms_32b.append(MmaAtomConfig(
            "SPARSE GMMA", 64, n, 16,
            DataType.tf32, asource,
            DataType.tf32, bsource,
            DataType.f32, MmaOperandSource.reg,
            128, True, False, 0, 90,
            kind=MmaKind.sparse))

  # iGMMA: s32 <- {s8|u8}
  gmma_atoms_int8  = []
  igmma_ab_types = [DataType.s8, DataType.u8]
  igmma_mods = [MmaPtxModifier.kNone, MmaPtxModifier.kSaturate]
  for atype in igmma_ab_types:
    for btype in igmma_ab_types:
      for asource in gmma_a_source_types:
        for n in sm90_gmma_n_sizes(atype, args.extended):
          for mod in igmma_mods:
            if not args.sparse:
              gmma_atoms_int8.append(MmaAtomConfig(
                  "GMMA", 64, n, 32,
                  atype, asource,
                  btype, bsource,
                  DataType.s32, MmaOperandSource.reg,
                  128, True, False, 0, 90, mod))
            else:
              gmma_atoms_int8.append(MmaAtomConfig(
                  "SPARSE GMMA", 64, n, 64,
                  atype, asource,
                  btype, bsource,
                  DataType.s32, MmaOperandSource.reg,
                  128, True, False, 0, 90, mod,
                  kind=MmaKind.sparse))

  # qGMMA - {fp16|fp32} <- {e4m3|e5m2}
  gmma_atoms_fp8  = []
  qgmma_ab_types = [DataType.e4m3, DataType.e5m2]
  qgmma_cd_types = [DataType.f16, DataType.f32]
  for atype in qgmma_ab_types:
    for btype in qgmma_ab_types:
      for n in sm90_gmma_n_sizes(atype, args.extended):
        for cbtype in qgmma_cd_types:
          for asource in gmma_a_source_types:
            if not args.sparse:
              gmma_atoms_fp8.append(MmaAtomConfig(
                  "GMMA", 64, n, 32,
                  atype, asource,
                  btype, bsource,
                  cbtype, MmaOperandSource.reg,
                  128, True, False, 0, 90))
            else:
              gmma_atoms_fp8.append(MmaAtomConfig(
                  "SPARSE GMMA", 64, n, 64,
                  atype, asource,
                  btype, bsource,
                  cbtype, MmaOperandSource.reg,
                  128, True, False, 0, 90,
                  kind=MmaKind.sparse))

  # bGMMA: s32 <- b1
  # bGMMA only supports one set of srcfmt, dstfmt, transA, transB, modifier
  # gmma_atoms_b1  = []
  # for asource in gmma_a_source_types:
  #   for n in sm90_gmma_n_sizes(DataType.b1, args.extended):
  #     gmma_atoms_b1.append(MmaAtomConfig(
  #         "GMMA", 64, n, 256,
  #         DataType.b1, asource,
  #         DataType.b1, bsource,
  #         DataType.s32, MmaOperandSource.reg,
  #         128, True, False, 0, 90, MmaPtxModifier.kAndPopc))

  # Emit to file
  namespace = "SM90::GMMA" + ("::SPARSE" if args.sparse else "")

  emit_cute_file_header(args.outfile)
  emit_namespace_header(args.outfile, namespace)
  emit_section_header(args.outfile)

  for atom in gmma_atoms_16b:
    emit_cute_sm90_16b_gmma_atom(args.outfile, atom)

  for atom in gmma_atoms_32b:
    emit_cute_sm90_32b_gmma_atom(args.outfile, atom)

  for atom in gmma_atoms_int8:
    emit_cute_sm90_int8_gmma_atom(args.outfile, atom)

  for atom in gmma_atoms_fp8:
    emit_cute_sm90_fp8_gmma_atom(args.outfile, atom)

  emit_section_footer(args.outfile)
  emit_namespace_footer(args.outfile, namespace)
  emit_cute_file_footer(args.outfile)
    
  if args.outfile is not sys.stdout:
    args.outfile.close()

  print("Generated {} GMMA MMA operators ({}, {} shapes)".format(
        len(gmma_atoms_int8) + len(gmma_atoms_16b) +
        len(gmma_atoms_32b) + len(gmma_atoms_fp8),
        "sparse" if args.sparse else "dense",
        "extended" if args.extended else "core"))
  return


def parse_cmd(argv: List[str]):
  parser = argparse.ArgumentParser(
      "Argument parser for SM90 MMA Atom Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which MMA atoms will be written.")
  parser.add_argument(
      '-s', '--sparse', action='store_true',
      help="Generate sparse MMA atoms instead of dense")
  parser.add_argument(
      '-e', '--extended', action='store_true',
      help="Generate extended GMMA instruction shapes instead of core shapes")
  return parser.parse_args(argv)


def main():
  args = parse_cmd(sys.argv[1:])
  generate_sm90_mma_atoms(args)


if __name__ == "__main__":
  main()
