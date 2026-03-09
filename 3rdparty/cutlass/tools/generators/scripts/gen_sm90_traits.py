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
from mma_atom_generator import make_mma_atom_name, sm90_gmma_n_sizes


def emit_cute_file_header(file):
  emit_license_header(file)
  header = """\
  
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits.hpp>

namespace cute {

"""
  file.write(header)

CUTE_GMMA_ATOM_NAME_TEMPLATE = "MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_GMMA_ATOM_NAME_TEMPLATE_16b = "MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"

CUTE_GMMA_ATOM_NAME_TEMPLATE_SPARSE = "GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_GMMA_ATOM_NAME_TEMPLATE_16b_SPARSE = "GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"

CUTE_GMMA_ATOM_NAME_ALIAS_TEMPLATE = "SM{sm_arch}_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_GMMA_ATOM_NAME_ALIAS_TEMPLATE_16b = "SM{sm_arch}_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"


CUTE_GMMA_ALIAS_TEMPLATE_01 = """\
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>"""

CUTE_GMMA_ALIAS_TEMPLATE_02 = """\
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>"""

CUTE_GMMA_ATOM_TRAITS_TEMPLATE_SS = """\
{alias_template}
using {template_alias_name} = SM90::GMMA::{mma_atom_name};

template <{template_mods_decl}>
struct MMA_Traits<{mma_atom_alias_name}>
{{
  using ValTypeD = {cd_val};
  using ValTypeA = {a_val};
  using ValTypeB = {b_val};
  using ValTypeC = {cd_val};

  using FrgTypeA = {smem_desc_a};
  using FrgTypeB = {smem_desc_b};

  using Shape_MNK = Shape<_{m},_{n},_{k}>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout<{m:>3},{k:>3}>;
  using BLayout = GMMA::ABLayout<{n:>3},{k:>3}>;
  using CLayout = GMMA::CLayout_{m}x{n};

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
}};
"""

CUTE_GMMA_ATOM_TRAITS_TEMPLATE_RS_16b = """\
{alias_template}
using {template_alias_name} = SM90::GMMA::{mma_atom_name};

template <{template_mods_decl}>
struct MMA_Traits<{mma_atom_alias_name}>
{{
  using ValTypeD = {cd_val};
  using ValTypeA = {a_val};
  using ValTypeE = {e_val};
  using ValTypeB = {b_val};
  using ValTypeC = {cd_val};

  using FrgTypeA = {smem_desc_a};
  using FrgTypeB = {smem_desc_b};

  using Shape_MNK = Shape<_{m},_{n},_{k}>;
  using ThrID   = Layout<_128>;

  using ALayout = GMMA::ALayout_{m}x{k};
  using BLayout = GMMA::ABLayout<{n:>3},{k:>3}>;
  using CLayout = GMMA::CLayout_{m}x{n};

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
}};
"""

CUTE_GMMA_ATOM_TRAITS_TEMPLATE_SS_SPARSE = """\
template <{template_mods_decl}>
struct MMA_Traits<{mma_atom_alias_name}>
{{
  using ValTypeD = {cd_val};
  using ValTypeA = {a_val};
  using ValTypeE = {e_val};
  using ValTypeB = {b_val};
  using ValTypeC = {cd_val};

  using FrgTypeA = {smem_desc_a};
  using FrgTypeB = {smem_desc_b};

  using Shape_MNK = Shape<_{m},_{n},_{k}>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout<{m:>3},{k:>3}>;
  using ELayout = GMMA::ELayout_{m}x{k};
  using BLayout = GMMA::ABLayout<{n:>3},{k:>3}>;
  using CLayout = GMMA::CLayout_{m}x{n};

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
}};
"""


CUTE_GMMA_ATOM_TRAITS_TEMPLATE_RS = """\
{alias_template}
using {template_alias_name} = SM90::GMMA::{mma_atom_name};

template <{template_mods_decl}>
struct MMA_Traits<{mma_atom_alias_name}>
{{
  using ValTypeD = {cd_val};
  using ValTypeA = {a_val};
  using ValTypeB = {b_val};
  using ValTypeC = {cd_val};

  using FrgTypeB = {smem_desc_b};

  using Shape_MNK = Shape<_{m},_{n},_{k}>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_{m}x{k};
  using BLayout = GMMA::ABLayout<{n:>3},{k:>3}>;
  using CLayout = GMMA::CLayout_{m}x{n};

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
}};
"""


CUTE_GMMA_ATOM_TRAITS_TEMPLATE_RS_SPARSE = """\
template <{template_mods_decl}>
struct MMA_Traits<{mma_atom_alias_name}>
{{
  using ValTypeD = {cd_val};
  using ValTypeA = {a_val};
  using ValTypeE = {e_val};
  using ValTypeB = {b_val};
  using ValTypeC = {cd_val};

  using FrgTypeB = {smem_desc_b};

  using Shape_MNK = Shape<_{m},_{n},_{k}>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_{m}x{k};
  using ELayout = GMMA::ELayout_{m}x{k};
  using BLayout = GMMA::ABLayout<{n:>3},{k:>3}>;
  using CLayout = GMMA::CLayout_{m}x{n};

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
}};
"""


def emit_cute_file_footer(file):
  footer = """\
} // end namespace cute
"""
  file.write(footer)


@post_separator
def emit_cute_sm90_gmma_traits(outfile, atom: MmaAtomConfig):
  if DataTypeSize[atom.atype] == 32:
    alias_template = CUTE_GMMA_ALIAS_TEMPLATE_01
    template_name = CUTE_GMMA_ATOM_NAME_TEMPLATE
    template_alias_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_SPARSE if atom.kind == MmaKind.sparse else CUTE_GMMA_ATOM_NAME_ALIAS_TEMPLATE
    template_mods = ["scaleA", "scaleB"]
    template_mods_decl = ["GMMA::ScaleIn scaleA", "GMMA::ScaleIn scaleB"]
    smem_desc_a = 'GMMA::smem_desc<GMMA::Major::K>'
    smem_desc_b = 'GMMA::smem_desc<GMMA::Major::K>'
  elif DataTypeSize[atom.atype] == 16:
    alias_template = CUTE_GMMA_ALIAS_TEMPLATE_02
    template_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_16b
    template_alias_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_16b_SPARSE if atom.kind == MmaKind.sparse else CUTE_GMMA_ATOM_NAME_ALIAS_TEMPLATE_16b
    template_mods = ["tnspA", "tnspB", "scaleA", "scaleB"]
    template_mods_decl = ["GMMA::Major tnspA", "GMMA::Major tnspB", "GMMA::ScaleIn scaleA", "GMMA::ScaleIn scaleB"]
    smem_desc_a = 'GMMA::smem_desc<tnspA>'
    smem_desc_b = 'GMMA::smem_desc<tnspB>'
  else:  # DataTypeSize[atom.atype] == 8:
    if atom.atype == DataType.e4m3 or atom.atype == DataType.e5m2: # FP8
      alias_template = CUTE_GMMA_ALIAS_TEMPLATE_01
      template_mods = ["scaleA", "scaleB"]
      template_mods_decl = ["GMMA::ScaleIn scaleA", "GMMA::ScaleIn scaleB"]
    else:
      # S8 or U8 GMMA does not have template arguments
      alias_template = ""
      template_mods = []
      template_mods_decl = []

    template_name = CUTE_GMMA_ATOM_NAME_TEMPLATE
    template_alias_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_SPARSE if atom.kind == MmaKind.sparse else CUTE_GMMA_ATOM_NAME_ALIAS_TEMPLATE

    # Major mode convention (K major is non-transposed, MN is transpose)
    smem_desc_a = 'GMMA::smem_desc<GMMA::Major::K>'
    smem_desc_b = 'GMMA::smem_desc<GMMA::Major::K>'

  if atom.kind == MmaKind.sparse:
    template_mods += ["spsel"]
    template_mods_decl += ["GMMA::SparseSel spsel"]

  # BLAS convention
  transA = 'T' if atom.transA else 'N'
  transB = 'T' if atom.transB else 'N'

  int_n = '_'+str(atom.n)
  int_k = '_'+str(atom.k)

  if atom.asource == MmaOperandSource.smem_desc:
    traits_template = (CUTE_GMMA_ATOM_TRAITS_TEMPLATE_SS_SPARSE
                       if atom.kind == MmaKind.sparse
                       else CUTE_GMMA_ATOM_TRAITS_TEMPLATE_SS)
  else:
    traits_template = (CUTE_GMMA_ATOM_TRAITS_TEMPLATE_RS_SPARSE
                       if atom.kind == MmaKind.sparse
                       else CUTE_GMMA_ATOM_TRAITS_TEMPLATE_RS)

  mma_atom_name = make_mma_atom_name(atom, template_name) + ("<{}>".format(", ".join(template_mods)) if len(template_mods) > 0 else "")
  mma_atom_alias_name = make_mma_atom_name(atom, template_alias_name) + ("<{}>".format(", ".join(template_mods)) if len(template_mods) > 0 else "")

  a_val  = DataTypeTag[atom.atype].split("cutlass::")[-1]
  b_val  = DataTypeTag[atom.btype].split("cutlass::")[-1]
  cd_val = DataTypeTag[atom.cdtype].split("cutlass::")[-1]
  e_val  = DataTypeTag[DataType.u8].split("cutlass::")[-1]

  if atom.kind == MmaKind.sparse:
      mma_atom_alias_name = "SM90::GMMA::SPARSE::{name}".format(arch=atom.sm_arch, name=mma_atom_alias_name)
      a_val = "sparse_elem<2, {}>".format(a_val)
      e_val = "sparse_elem<{}, {}>".format(4 if atom.atype == DataType.tf32 else 8, e_val)
  else:
    e_val = ""

  outfile.write(traits_template.format(
      alias_template = alias_template,
      template_alias_name = make_mma_atom_name(atom,template_alias_name),
      mma_atom_name = mma_atom_name,
      mma_atom_alias_name = mma_atom_alias_name,
      atype=str.upper(DataTypeNames[atom.atype]),
      btype=str.upper(DataTypeNames[atom.btype]),
      cdtype=str.upper(DataTypeNames[atom.cdtype]),
      a_val=a_val,
      e_val=e_val,
      b_val=b_val,
      cd_val=cd_val,
      transA=transA,
      transB=transB,
      m=atom.m,
      n=atom.n,
      k=atom.k,
      int_n=int_n,
      int_k=int_k,
      smem_desc_a=smem_desc_a,
      smem_desc_b=smem_desc_b,
      template_mods_decl=", ".join(template_mods_decl)
  ))


def generate_sm90_mma_atoms(args: argparse.Namespace):

  gmma_atoms = []
  gmma_a_source_types = [MmaOperandSource.smem_desc, MmaOperandSource.reg]
  bsource = MmaOperandSource.smem_desc

  # hGMMA::: {fp32|fp16} <- {fp16|bf16}
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
            gmma_atoms.append(MmaAtomConfig(
                "hGMMA::",
                64, n, 16,
                ab_type, asource,
                ab_type, bsource,
                cd_type, MmaOperandSource.reg,
                128, None, None, 0, 90))
          else:
            gmma_atoms.append(MmaAtomConfig(
                "hGMMA::SPARSE::",
                64, n, 32,
                ab_type, asource,
                ab_type, bsource,
                cd_type, MmaOperandSource.reg,
                128, None, None, 0, 90,
                kind=MmaKind.sparse))

  # hGMMA:: - fp32 <- tf32
  # tf32 only supports a single set of dstfmt, transA, transB
  for n in sm90_gmma_n_sizes(DataType.tf32, args.extended):
    for asource in gmma_a_source_types:
      if not args.sparse:
        gmma_atoms.append(MmaAtomConfig(
            "hGMMA::", 64, n, 8,
            DataType.tf32, asource,
            DataType.tf32, bsource,
            DataType.f32, MmaOperandSource.reg,
            128, True, False, 0, 90))
      else:
        gmma_atoms.append(MmaAtomConfig(
            "hGMMA::SPARSE::", 64, n, 16,
            DataType.tf32, asource,
            DataType.tf32, bsource,
            DataType.f32, MmaOperandSource.reg,
            128, True, False, 0, 90,
            kind=MmaKind.sparse))

  # iGMMA::: s32 <- {s8|u8}
  igmma_ab_types = [DataType.s8, DataType.u8]
  igmma_mods = [MmaPtxModifier.kNone, MmaPtxModifier.kSaturate]
  for atype in igmma_ab_types:
    for btype in igmma_ab_types:
      for asource in gmma_a_source_types:
        for n in sm90_gmma_n_sizes(atype, args.extended):
          for mod in igmma_mods:
            if not args.sparse:
              gmma_atoms.append(MmaAtomConfig(
                  "iGMMA::", 64, n, 32,
                  atype, asource,
                  btype, bsource,
                  DataType.s32, MmaOperandSource.reg,
                  128, True, False, 0, 90, mod))
            else:
              gmma_atoms.append(MmaAtomConfig(
                  "iGMMA::SPARSE::", 64, n, 64,
                  atype, asource,
                  btype, bsource,
                  DataType.s32, MmaOperandSource.reg,
                  128, True, False, 0, 90, mod,
                  kind=MmaKind.sparse))

  # qGMMA - {fp16|fp32} <- {e4m3|e5m2}
  qgmma_ab_types = [DataType.e4m3, DataType.e5m2]
  qgmma_cd_types = [DataType.f16, DataType.f32]
  for atype in qgmma_ab_types:
    for btype in qgmma_ab_types:
      for n in sm90_gmma_n_sizes(atype, args.extended):
        for cbtype in qgmma_cd_types:
          for asource in gmma_a_source_types:
            if not args.sparse:
              gmma_atoms.append(MmaAtomConfig(
                  "qGMMA::", 64, n, 32,
                  atype, asource,
                  btype, bsource,
                  cbtype, MmaOperandSource.reg,
                  128, True, False, 0, 90))
            else:
              gmma_atoms.append(MmaAtomConfig(
                  "qGMMA::SPARSE::", 64, n, 64,
                  atype, asource,
                  btype, bsource,
                  cbtype, MmaOperandSource.reg,
                  128, True, False, 0, 90,
                  kind=MmaKind.sparse))

  # Emit to file
  emit_cute_file_header(args.outfile)
  emit_section_header(args.outfile)

  for atom in gmma_atoms:
    emit_cute_sm90_gmma_traits(args.outfile, atom)

  emit_section_footer(args.outfile)
  emit_cute_file_footer(args.outfile)

  if args.outfile is not sys.stdout:
    args.outfile.close()

  print("Generated {} GMMA MMA traits ({}, {} shapes)".format(
        len(gmma_atoms),
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
