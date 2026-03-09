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
import itertools
from typing import *
from generator_library import *
from mma_atom_generator import make_mma_atom_name, sm90_shape_guard, sm90_gmma_n_sizes_all

CUTE_GMMA_ATOM_NAME_TEMPLATE = "SM90::GMMA::MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_GMMA_ATOM_NAME_TEMPLATE_16b = "SM90::GMMA::MMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"
CUTE_GMMA_ATOM_NAME_TEMPLATE_SPARSE = "SM90::GMMA::SPARSE::GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}_TN"
CUTE_GMMA_ATOM_NAME_TEMPLATE_16b_SPARSE = "SM90::GMMA::SPARSE::GMMA_{m}x{n}x{k}_{cdtype}{atype}{btype}_{asrc}{bsrc}"


def emit_cute_file_header(file):
  emit_license_header(file)
  header = """\
  
#include <cute/arch/mma_sm90.hpp>

namespace cute {
namespace SM90::GMMA {

"""
  file.write(header)


def emit_cute_file_footer(file):
  footer = """\
} // namespace SM90::GMMA
} // namespace cute
"""
  file.write(footer)


@sm90_shape_guard
def emit_case_n(file, atom: MmaAtomConfig, case_index: int):
  if DataTypeSize[atom.atype] == 16:
    template_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_16b_SPARSE if atom.kind == MmaKind.sparse else CUTE_GMMA_ATOM_NAME_TEMPLATE_16b
    template_args = ["MajorA", "MajorB", "Args..."]
  else:
    template_name = CUTE_GMMA_ATOM_NAME_TEMPLATE_SPARSE if atom.kind == MmaKind.sparse else CUTE_GMMA_ATOM_NAME_TEMPLATE
    template_args = ["Args..."]
    if (atom.atype == DataType.s8 or atom.atype == DataType.u8) and atom.kind == MmaKind.dense:
      template_args = []

  case = """\
      {else_clause}if constexpr (Tile_N % {inst_n} == 0) {{
        return {op_name}{op_args}{{}};
      }}
""".format(
    inst_n=atom.n,
    op_name=make_mma_atom_name(atom, template_name),
    op_args=("<{}>".format(", ".join(template_args))) if len(template_args) > 0 else "",
    else_clause = "else " if case_index > 0 else ""
    )
  file.write(case)


def emit_case_dtype_ab(file, atoms: Sequence[MmaAtomConfig], case_index: int):
  atoms = sorted(atoms, key=lambda atom: atom.n, reverse=True)
  atom = atoms[0]
  assert all([a.atype==atom.atype and a.btype==atom.btype for a in atoms])

  header = """\

    // Input A: {atype} ; Input B: {btype}
    {else_clause}if constexpr (is_same_v<ElementA, {atype}> && is_same_v<ElementB, {btype}>) {{
""".format(
    atype=DataTypeTag[atom.atype].split("cutlass::")[-1],
    btype=DataTypeTag[atom.btype].split("cutlass::")[-1],
    else_clause = "else " if case_index > 0 else ""
  )

  if DataTypeSize[atom.atype] != 16:
    header += """\
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
"""

  header += """\
      static_assert(size<2>(TileShape_MNK{{}}) % {alignk} == 0, "Tile_K must be a multiple of {alignk}.");

""".format(alignk=atom.k)
  
  footer = """\
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }
"""
  
  file.write(header)
  for i, a in enumerate(atoms):
    emit_case_n(file, a, i)
  file.write(footer)


def emit_case_dtype_cd(file, atoms: Sequence[MmaAtomConfig], case_index: int):
  # Sort order that tries to match bespoke code as close as possible
  sort_order = {
    DataType.f16: 1,
    DataType.bf16: 2,
    DataType.tf32: 3,
    DataType.s8: 4,
    DataType.u8: 5,
    DataType.e4m3: 4,
    DataType.e5m2: 5,
  }
  atoms = sorted(atoms, key=lambda a: (sort_order[a.atype], sort_order[a.btype]))
  atom = atoms[0]
  assert all([a.cdtype==atom.cdtype for a in atoms])

  header = """\

  // {cdtype_name} accumulator
  {else_clause}if constexpr (is_same_v<ElementC, {cdtype}>) {{
""".format(
    cdtype=DataTypeTag[atom.cdtype].split("cutlass::")[-1],
    cdtype_name=str.upper(DataTypeNames[atom.cdtype]),
    else_clause = "else " if case_index > 0 else ""
  )
  footer = """\

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }
"""
  
  file.write(header)
  for i, (ab_types, atom_group) in enumerate(itertools.groupby(atoms, key=lambda a: (a.atype, a.btype))):
    emit_case_dtype_ab(file, atom_group, i)
  file.write(footer)


def emit_selector(file, atoms: Sequence[MmaAtomConfig]):
  # Sort order that tries to match bespoke code as close as possible
  sort_order = {
    DataType.f16: 1,
    DataType.f32: 2,
    DataType.s32: 3,
  }
  atoms = sorted(atoms, key=lambda a: sort_order[a.cdtype])
  atom = atoms[0]
  assert all([a.asource == atom.asource and a.kind==atom.kind for a in atoms])
  
  header = """\
template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K,
  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
CUTE_HOST_DEVICE constexpr
auto
{src}_op_selector{sparse}()
{{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{{}}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{{}}) % 64 == 0, "Tile_M must be a multiple of 64.");
""".format(
    src = "rs" if atom.asource == MmaOperandSource.reg else "ss",
    sparse = "_sparse" if atom.kind == MmaKind.sparse else ""
  )

  if atom.asource == MmaOperandSource.reg:
    header += """\
  static_assert(MajorA == GMMA::Major::K, "Register source A operand GMMAs must have K-major A layout.");
"""
  header += """\
  auto Tile_N = size<1>(TileShape_MNK{});
"""

  footer = """\

  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

"""

  file.write(header)
  for i, (cd_type, atom_group) in enumerate(itertools.groupby(atoms, key=lambda a: a.cdtype)):
    emit_case_dtype_cd(file, atom_group, i)
  file.write(footer)


def emit_all_selectors(file, atoms: Sequence[MmaAtomConfig]):
  # Sort order that tries to match bespoke code as close as possible
  src_sort_order = {
    MmaOperandSource.smem_desc: 1,
    MmaOperandSource.reg: 2,
  }
  kind_sort_order = {
    MmaKind.dense: 1,
    MmaKind.sparse: 2,
  }
  atoms = sorted(atoms, key=lambda a: (src_sort_order[a.asource], kind_sort_order[a.kind]))
  for kind_src, atom_group in itertools.groupby(atoms, key=lambda a: (a.kind.value, a.asource.value)):
    emit_selector(file, atom_group)


def generate_sm90_mma_selectors(args: argparse.Namespace):
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
      for n in sm90_gmma_n_sizes_all(ab_type):
        for asource in gmma_a_source_types:
          gmma_atoms.append(MmaAtomConfig(
              "hGMMA::",
              64, n, 16,
              ab_type, asource,
              ab_type, bsource,
              cd_type, MmaOperandSource.reg,
              128, None, None, 0, 90))
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
  for n in sm90_gmma_n_sizes_all(DataType.tf32):
    for asource in gmma_a_source_types:
      gmma_atoms.append(MmaAtomConfig(
          "hGMMA::", 64, n, 8,
          DataType.tf32, asource,
          DataType.tf32, bsource,
          DataType.f32, MmaOperandSource.reg,
          128, True, False, 0, 90))
      gmma_atoms.append(MmaAtomConfig(
          "hGMMA::SPARSE::", 64, n, 16,
          DataType.tf32, asource,
          DataType.tf32, bsource,
          DataType.f32, MmaOperandSource.reg,
          128, True, False, 0, 90,
          kind=MmaKind.sparse))

  # iGMMA::: s32 <- {s8|u8}
  igmma_ab_types = [DataType.s8, DataType.u8]
  for atype in igmma_ab_types:
    for btype in igmma_ab_types:
      for asource in gmma_a_source_types:
        for n in sm90_gmma_n_sizes_all(atype):
          gmma_atoms.append(MmaAtomConfig(
              "iGMMA::", 64, n, 32,
              atype, asource,
              btype, bsource,
              DataType.s32, MmaOperandSource.reg,
              128, True, False, 0, 90))
          gmma_atoms.append(MmaAtomConfig(
              "iGMMA::SPARSE::", 64, n, 64,
              atype, asource,
              btype, bsource,
              DataType.s32, MmaOperandSource.reg,
              128, True, False, 0, 90,
              kind=MmaKind.sparse))

  # qGMMA - {fp16|fp32} <- {e4m3|e5m2}
  qgmma_ab_types = [DataType.e4m3, DataType.e5m2]
  qgmma_cd_types = [DataType.f16, DataType.f32]
  for atype in qgmma_ab_types:
    for btype in qgmma_ab_types:
      for n in sm90_gmma_n_sizes_all(atype):
        for cbtype in qgmma_cd_types:
          for asource in gmma_a_source_types:
            gmma_atoms.append(MmaAtomConfig(
                "qGMMA::", 64, n, 32,
                atype, asource,
                btype, bsource,
                cbtype, MmaOperandSource.reg,
                128, True, False, 0, 90))
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

  emit_all_selectors(args.outfile, gmma_atoms)

  emit_section_footer(args.outfile)
  emit_cute_file_footer(args.outfile)

  if args.outfile is not sys.stdout:
    args.outfile.close()

  print("Generated {} GMMA op selector cases".format(len(gmma_atoms)))
  return


def parse_cmd(argv: List[str]):
  parser = argparse.ArgumentParser(
      "Argument parser for SM90 MMA Atom Generator.")
  parser.add_argument(
      '-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
      help="Path to the output file to which MMA atoms will be written.")
  return parser.parse_args(argv)


def main():
  args = parse_cmd(sys.argv[1:])
  generate_sm90_mma_selectors(args)


if __name__ == "__main__":
  main()
