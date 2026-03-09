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
from typing import *
from generator_library import *

sm90_dmma_instruction_template = "mma.sync.aligned.m{m}n{n}k{k}.row.col.f64.f64.f64.f64"
sm90_gmma_instruction_template = "wgmma.mma_async{spmod}.sync.aligned.m{m}n{n}k{k}.{cdtype}.{atype}.{btype}{modifier}"
sm80_mma_instruction_template  = "mma.sync.aligned.m{m}n{n}k{k}{transA}{transB}.{cdtype}.{atype}.{btype}.{cdtype}{modifier}"
sm120_mma_instruction_template  = "_mma.m{m}n{n}k{k}{transA}{transB}.{cdtype}.{atype}.{btype}.{cdtype}{modifier}"
sm120_blockscaled_mma_instruction_template  = "_mma.block_scale.scale_vec::{vec}X.m{m}n{n}k{k}{transA}{transB}.{cdtype}.{atype}.{btype}.{cdtype}.{sftype}{modifier}"


def sm90_gmma_n_sizes_all(dtype: DataType):
  if dtype in [DataType.s8, DataType.u8, DataType.b1]:
    return [8, 16, 24] + list(range(32, 256+1, 16))
  else:
    return list(range(8, 256+1, 8))


def sm90_gmma_n_sizes(dtype: DataType, extended: bool):
  sm90_gmma_n_sizes_core = [8, 16, 32, 64, 96, 128, 192, 256]
  if extended:
    return [n for n in sm90_gmma_n_sizes_all(dtype) if n not in sm90_gmma_n_sizes_core]
  else:
    return sm90_gmma_n_sizes_core


def sm90_shape_guard(function):
  def wrapped(outfile, atom: MmaAtomConfig, *args):
    guard = atom.sm_arch == 90 and atom.threads == 128 and atom.n not in sm90_gmma_n_sizes(atom.atype, False)
    if guard:
      outfile.write("""\
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
""")
    function(outfile, atom, *args)
    if guard:
          outfile.write("""\
#endif
""")
  return wrapped


def make_mma_atom_reg_array_type(
        arg_type: DataType,
        arg_source: MmaOperandSource,
        arg_count: int) -> str:
  if arg_source == MmaOperandSource.smem_desc:
    arg_reg_type = "uint64_t"
  else:
    arg_reg_type = DataTypeTag[DataTypeBackingStore[arg_type]]
  return arg_reg_type + "[{}]".format(arg_count)


def get_ptx_template(atom: MmaAtomConfig, is_external=True) -> str:
  if atom.sm_arch == 80:
    assert is_external, "no"
    return sm80_mma_instruction_template
  elif atom.sm_arch == 120:
    if sum(atom.blockscale_vecsize) > 0:
      return sm120_blockscaled_mma_instruction_template
    else:
      return sm120_mma_instruction_template

  if atom.sm_arch == 90 and atom.threads == 32:
    return sm90_dmma_instruction_template
  if atom.sm_arch == 90 and atom.threads == 128:
    return sm90_gmma_instruction_template
  raise NotImplementedError("Generator does not support this PTX instruction.")


# Override datatype name based on manual modifications made in the CuTe source,
# or return the default datatype name
def possibly_overridden_name(dtype: DataType) -> str:
  overridden_types = {
    DataType.b1: "u1"
  }
  return overridden_types.get(dtype, DataTypeNames[dtype])


# Generic atom formatter; used for MMA wrapper struct type name and header comment
def generic_mma_atom_format(template, atom):
  return template.format(
      sm_arch=atom.sm_arch,
      mma_tag=atom.tag,
      m=atom.m,
      n=atom.n,
      k=atom.k,
      atype=possibly_overridden_name(atom.atype).upper(),
      btype=possibly_overridden_name(atom.btype).upper(),
      cdtype=possibly_overridden_name(atom.cdtype).upper(),
      sftype=possibly_overridden_name(atom.sftype).upper(),
      transA='T' if atom.transA else 'N',
      transB='T' if atom.transB else 'N',
      asrc='S' if atom.asource == MmaOperandSource.smem_desc else 'R',
      bsrc='S' if atom.bsource == MmaOperandSource.smem_desc else 'R')


def add_modifiers_to_atom_name(atom_name: str, atom: MmaAtomConfig):
  mod = atom.modifier
  if mod != MmaPtxModifier.kNone:
    mod_name = ((str(mod).split('.')[1])[1:]).upper()
    atom_name += '_' + mod_name
  return atom_name

# Formatter for the PTX instruction itself


def make_mma_atom_instruction(instruction_template: str, atom: MmaAtomConfig):
  # although we do not know the template, format ignores any kwargs that are not used
  if atom.sm_arch == 90:
    transA = ".transA" if not atom.transA else ""
    transB = ".transB" if atom.transB else ""
  else:
    transA = ".row" if atom.transA else ".col"
    transB = ".row" if atom.transB else ".col"

  format_params = {
      "m": atom.m,
      "n": atom.n,
      "k": atom.k,
      "transA": transA,
      "transB": transB,
      "cdtype": DataTypeNames[atom.cdtype],
      "atype": DataTypeNames[atom.atype],
      "btype": DataTypeNames[atom.btype],
      "modifier": MmaPtxModifierToPtxStr[atom.modifier],
      "vec": atom.k // atom.blockscale_vecsize[0] if atom.blockscale_vecsize[0] > 0 else 0,
      "sftype": DataTypeNames[atom.sftype],
      "spmod" : ".sp" if atom.kind == MmaKind.sparse else "",
  }

  return "\"" + instruction_template.format(**format_params) + " \""

def make_blockscaled_mma_atom_instruction(instruction_template: str, atom: MmaAtomConfig, vec_size: int):
  transA = ".row" if atom.transA else ".col"
  transB = ".row" if atom.transB else ".col"

  format_params = {
      "m": atom.m,
      "n": atom.n,
      "k": atom.k,
      "transA": transA,
      "transB": transB,
      "cdtype": DataTypeNames[atom.cdtype],
      "atype": DataTypeNames[atom.atype],
      "btype": DataTypeNames[atom.btype],
      "modifier": MmaPtxModifierToPtxStr[atom.modifier],
      "vec": atom.k // vec_size,
      "sftype": DataTypeNames[atom.sftype],
      "spmod" : ".sp" if atom.kind == MmaKind.sparse else "",
  }
  return "\"" + instruction_template.format(**format_params) + " \""


# Generates the PTX wrapper fma input names : void fma(...generate...)
def make_mma_atom_fma_exploded_args(
        atom: MmaAtomConfig,
        cd_reg_count: int,
        a_reg_count: int,
        b_reg_count: int,
        fma_args_indent: int) -> str:
  def make_one_args_str(io_name: str, io_type: str, io_width: int, idx_width: int, count: int, indent: int, final_args: bool=False):
    io_template = "{io_type:{io_width}}& {io_name}{idx:0{idx_width}}{comma}"
    io_str = ""
    for i in range(count):
      comma = "" if final_args and i == count - 1 else ","
      io_str += io_template.format(
          io_type=io_type, io_width=io_width, io_name=io_name, idx=i, idx_width=idx_width, comma=comma)
      if not final_args or i + 1 != count:
        if i > 0 and (i + 1) % 4 == 0:
          io_str += "\n"
        else:
          io_str += ' '
    # if we do not have enough regs to fill a row, make sure we end the current
    # line after stripping out the trailing space
    if count % 4 != 0 and not final_args:
      io_str = io_str[:-1]
      io_str += "\n"
    return io_str

  # figure out input types first
  d_args_type = DataTypeTag[DataTypeBackingStore[atom.cdtype]]
  c_args_type = DataTypeTag[DataTypeBackingStore[atom.cdtype]] + " const"
  sfa_args_type = DataTypeTag[DataTypeBackingStore[atom.sftype]] + " const"
  sfb_args_type = DataTypeTag[DataTypeBackingStore[atom.sftype]] + " const"

  if atom.asource == MmaOperandSource.smem_desc:
    a_args_type = "uint64_t const"
  else:
    a_args_type = DataTypeTag[DataTypeBackingStore[atom.atype]] + " const"
  if atom.bsource == MmaOperandSource.smem_desc:
    b_args_type = "uint64_t const"
  else:
    b_args_type = DataTypeTag[DataTypeBackingStore[atom.btype]] + " const"

  # calculate widths for padding so everthing is vertically aligned
  io_type_width = max(len(d_args_type), len(a_args_type),
                      len(b_args_type), len(c_args_type))
  idx_width = len(str(max(cd_reg_count, b_reg_count, a_reg_count)))

  # make strings
  d_args_str = make_one_args_str(
      'd', d_args_type, io_type_width, idx_width, cd_reg_count, fma_args_indent)

  if atom.asource == MmaOperandSource.smem_desc:
    a_args_str = "uint64_t const& desc_a,\n"
  else:
    a_args_str = make_one_args_str(
        'a', a_args_type, io_type_width, idx_width, a_reg_count, fma_args_indent)

  if atom.bsource == MmaOperandSource.smem_desc:
    b_args_str = "uint64_t const& desc_b,\n"
  else:
    b_args_str = make_one_args_str(
        'b', b_args_type, io_type_width, idx_width, b_reg_count, fma_args_indent)

  # For SM80-style kernels, C args are the final arguments
  final_args = atom.sm_arch == 80 or (atom.sm_arch == 120 and sum(atom.blockscale_vecsize) == 0)
  c_args_str = make_one_args_str(
      'c', c_args_type, io_type_width, idx_width, cd_reg_count, fma_args_indent, final_args)

  sfa_args_str = make_one_args_str(
      'sfa', sfa_args_type, io_type_width, idx_width, 1, fma_args_indent, False)

  sfb_args_str = make_one_args_str(
      'sfb', sfb_args_type, io_type_width, idx_width, 1, fma_args_indent, True)
  
  if atom.kind == MmaKind.sparse and atom.sm_arch in [80, 90]:
    e_args_str = DataTypeTag[DataTypeBackingStore[DataType.u32]] + " const& e,\n"
  else:
    e_args_str = ""

  # Deal with scaleD
  scale_d_args_str = "GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One"
    
  # Deal with WGMMA having in place accumulation
  if (atom.sm_arch == 90 and atom.threads == 128):
    # make full string
    io_str = a_args_str + b_args_str + d_args_str + e_args_str + scale_d_args_str
  elif (atom.sm_arch == 120 and sum(atom.blockscale_vecsize) > 0):
    # make full string
    io_str = d_args_str + a_args_str + b_args_str + c_args_str + sfa_args_str + sfb_args_str
  else:
    # make full string
    io_str = d_args_str + a_args_str + b_args_str + c_args_str
    if not final_args:
      io_str += scale_d_args_str
 
  # add indentation
  return io_str.replace('\n', '\n'.ljust(fma_args_indent + 1))


# Makes a PTX operand string starting at index start. Takes care of padding
def make_ptx_operands(
        start: int,
        stop: int,
        max_operand_width: int,
        wraparound: int,
        trailing_str: str,
        indent: int) -> str:
  # counter just checks if we should add a newline or not
  # cannot do this with index i because start not always 0
  if not indent or indent < 0:
    indent = 0
  counter = 0
  ptx = "\"{"
  for i in range(start, stop):
    if counter > 0 and counter % wraparound == 0:
      ptx += '\"\n{}" '.format(''.ljust(indent))

    tmp = "%{}".format(i)
    # add ',' and align for '%' if not the last one
    if i < stop - 1:
      tmp = (tmp + ',').ljust(max_operand_width)
    ptx += tmp
    counter += 1

  # finish this block of bindings, and cap off with a trailing string (usually ',')
  ptx += "}}{}\"".format(trailing_str)
  return ptx


def make_wgmma_mod_ptx_operands(
        atom: MmaAtomConfig,
        start: int,
        max_operand_width: int,
        wraparound: int,
        trailing_str: str,
        indent: int) -> str:
  assert(atom.sm_arch == 90 and atom.threads == 128)
  mod_count = 0
  if DataTypeSize[atom.atype] == 16 and atom.asource == MmaOperandSource.smem_desc:
    mod_count = 5
  elif DataTypeSize[atom.atype] == 16 and atom.asource == MmaOperandSource.reg:
    mod_count = 4
  elif DataTypeSize[atom.atype] == 32:
    mod_count = 3
  else:  # DataTypeSize[atom.atype] == 8:
    if atom.atype == DataType.e4m3 or atom.atype == DataType.e5m2: # FP8
      mod_count = 3
    else: # DataType.s8
      mod_count = 1

  stop = start + mod_count
  ptx = "\" "
  for i in range(start, stop):
    if i == start:
      tmp = "p" # scaleD is a predicate.
    else:
      tmp = "%{}".format(i)
    # add ',' and align for '%' if not the last one
    if i < stop - 1:
      tmp = (tmp + ',').ljust(max_operand_width)
    ptx += tmp
  ptx += "{}\"".format(trailing_str)
  return ptx


def make_ptx_bindings(
        atom: MmaAtomConfig,
        cd_reg_count: int,
        a_reg_count: int,
        b_reg_count: int,
        ptx_indent) -> str:
  def make_one_ptx_binding(constrain: str, constrain_width: str, io_name: str, idx_w: int, count: int, delimit: bool):
    binding_template = "{constrain:>{constrain_width}}({io_name}{idx:0{idx_width}}),"
    binding_str = ": " if delimit else "  "
    for i in range(count):
      binding_str += binding_template.format(
          constrain=constrain, constrain_width=constrain_width, io_name=io_name, idx=i, idx_width=idx_w)
      if i > 0 and i < count + 1 and (i + 1) % 4 == 0:
        binding_str += "\n  "
      else:
        binding_str += ' '
    # if we do not have enough regs to fill a row, make sure we end the current
    # line after stripping out the trailing newline and two spaces
    if (count % 4 != 0):
      binding_str = binding_str[:-3]
      binding_str += "),\n  "

    # remove the two trailing spaces after newline
    binding_str = binding_str[:-2]
    return binding_str

  d_args_name = 'd'
  if (atom.sm_arch == 90 and atom.threads == 128):
    d_constrain_type = "\"+" + \
        DataTypePtxConstrain[DataTypeBackingStore[atom.cdtype]] + "\""
  else:
    d_constrain_type = "\"=" + \
        DataTypePtxConstrain[DataTypeBackingStore[atom.cdtype]] + "\""

  c_args_name = 'c'
  c_constrain_type = "\"" + \
      DataTypePtxConstrain[DataTypeBackingStore[atom.cdtype]] + "\""

  a_args_name = 'a'
  a_constrain_type = "\"" + \
      DataTypePtxConstrain[DataTypeBackingStore[atom.atype]] + "\""

  b_args_name = 'b'
  b_constrain_type = "\"" + \
      DataTypePtxConstrain[DataTypeBackingStore[atom.btype]] + "\""

  sfa_args_name = 'sfa'
  sfa_constrain_type = "\"" + \
      DataTypePtxConstrain[DataTypeBackingStore[atom.sftype]] + "\""

  sfb_args_name = 'sfb'
  sfb_constrain_type = "\"" + \
      DataTypePtxConstrain[DataTypeBackingStore[atom.sftype]] + "\""


  # figure out ptx constrain type
  constrain_width = max(len(d_constrain_type), len(a_constrain_type),
                        len(b_constrain_type), len(c_constrain_type))

  # calculate widths for padding so everthing is vertically aligned
  idx_width = len(str(max(cd_reg_count, b_reg_count, a_reg_count)))

  if sum(atom.blockscale_vecsize) > 0:
    constrain_width += 1
    
  # make strings
  d_binding = make_one_ptx_binding(
      d_constrain_type, constrain_width, d_args_name, idx_width, cd_reg_count, True)
  # replace trailing ',\n' with only a '\n' to remove the comma
  d_binding = d_binding[:-2] + "\n"

  # we would rather not hardcode this, but we don't want index suffix on descriptors
  if atom.asource == MmaOperandSource.smem_desc:
    a_binding = ": {constrain:>{constrain_width}}({io_name}),\n".format(
        constrain="\"l\"", constrain_width=constrain_width, io_name="desc_a")
  else:
    a_binding = make_one_ptx_binding(
        a_constrain_type, constrain_width, a_args_name, idx_width, a_reg_count, True)

  if atom.bsource == MmaOperandSource.smem_desc:
    b_binding = "  {constrain:>{constrain_width}}({io_name}),\n".format(
        constrain="\"l\"", constrain_width=constrain_width, io_name="desc_b")
  else:
    b_binding = make_one_ptx_binding(
        b_constrain_type, constrain_width, b_args_name, idx_width, b_reg_count, False)

  c_binding = make_one_ptx_binding(
      c_constrain_type, constrain_width, c_args_name, idx_width, cd_reg_count, False)
  
  if atom.kind == MmaKind.sparse and atom.sm_arch in [80, 90]:
    e_binding  = ''.ljust(ptx_indent - 3) + "\"r\"(e), \"n\"(int32_t(spsel)),\n"
  else:
    e_binding = ""

  sfa_binding = make_one_ptx_binding(
      sfa_constrain_type, constrain_width, sfa_args_name, 1, 1, False)[:-1] + " \"h\"(bidA), \"h\"(tidA),\n"
  sfb_binding = make_one_ptx_binding(
      sfb_constrain_type, constrain_width, sfb_args_name, 1, 1, False)[:-1] + " \"h\"(bidB), \"h\"(tidB)"

  if (atom.sm_arch == 90 and atom.threads == 128):
    # remove trailing ',\n', no replacement here
    mod_bindings = make_wgmma_mod_ptx_bindings(atom, ptx_indent)
    # make full string
    bindings = d_binding + a_binding + b_binding + e_binding + mod_bindings
  elif (atom.sm_arch == 120 and sum(atom.blockscale_vecsize) > 0):
    # make full string
    bindings = d_binding + a_binding + b_binding + c_binding + sfa_binding + sfb_binding
  else:
    # remove trailing ',\n', no replacement here
    c_binding = c_binding[:-2]
    # make full string
    bindings = d_binding + a_binding + b_binding + c_binding
  # add indentation
  return bindings.replace('\n', '\n'.ljust(ptx_indent + 1))


def make_wgmma_mod_ptx_bindings(atom: MmaAtomConfig, ptx_indent: int):
  assert(atom.sm_arch == 90 and atom.threads == 128)
  if DataTypeSize[atom.atype] == 16 and atom.asource == MmaOperandSource.smem_desc:
    mod_bindings = "\"r\"(int32_t(scale_D)), \"n\"(int32_t(scaleA)), \"n\"(int32_t(scaleB)), \"n\"(int32_t(tnspA)), \"n\"(int32_t(tnspB))"
  elif DataTypeSize[atom.atype] == 16 and atom.asource == MmaOperandSource.reg:
    mod_bindings = "\"r\"(int32_t(scale_D)), \"n\"(int32_t(scaleA)), \"n\"(int32_t(scaleB)), \"n\"(int32_t(tnspB))"
  elif DataTypeSize[atom.atype] == 32:
    mod_bindings = "\"r\"(int32_t(scale_D)), \"n\"(int32_t(scaleA)), \"n\"(int32_t(scaleB))"
  else:  # DataTypeSize[atom.atype] == 8
    if atom.atype == DataType.e4m3 or atom.atype == DataType.e5m2: # FP8 
      mod_bindings = "\"r\"(int32_t(scale_D)), \"n\"(int32_t(scaleA)), \"n\"(int32_t(scaleB))"
    else: #  DataType.s8
      mod_bindings = "\"r\"(int32_t(scale_D))"
  return ''.ljust(ptx_indent - 3) + mod_bindings


def make_mma_atom_name(atom: MmaAtomConfig, template: str):
  return add_modifiers_to_atom_name(
      generic_mma_atom_format(template, atom), atom)


def emit_mma_atom_wrapper(
        file,
        mma_atom_header_template: str,
        mma_atom_name_template: str,
        mma_atom_wrapper_template: str,
        atom: MmaAtomConfig,
        fma_args_indent: int,
        ptx_indent: int,
        round_up_ab_sizes: bool = False):
  sm_arch = atom.sm_arch
  assert sm_arch in [80, 90, 120], "Only SM80, SM90, and SM120 are supported for now."
  if sm_arch == 90 and atom.threads == 128 and atom.asource == MmaOperandSource.reg:
    assert atom.transA == None or atom.transA == True

  # number of values held by each thread is values in the block div by threads in atom
  cd_val_count = (atom.m * atom.n) // atom.threads
  a_val_count = (atom.m * atom.k) // atom.threads
  b_val_count = (atom.n * atom.k) // atom.threads

  # number of actual registers is values times width of each value div by width of backing
  cd_reg_count = (cd_val_count * DataTypeSize[atom.cdtype]
                  ) // DataTypeSize[DataTypeBackingStore[atom.cdtype]]

  a_size = DataTypeSize[atom.atype]
  b_size = DataTypeSize[atom.btype]
  if round_up_ab_sizes:
    a_size = ((a_size + 7) // 8) * 8
    b_size = ((b_size + 7) // 8) * 8

  # unless the operands come from a shared memory descriptor
  if atom.asource == MmaOperandSource.smem_desc:
    a_reg_count = 1
  else:
    a_reg_count = (a_val_count * a_size
                   ) // DataTypeSize[DataTypeBackingStore[atom.atype]]
    if atom.kind == MmaKind.sparse:
      a_reg_count = a_reg_count // 2

  if atom.bsource == MmaOperandSource.smem_desc:
    b_reg_count = 1
  else:
    b_reg_count = (b_val_count * b_size
                   ) // DataTypeSize[DataTypeBackingStore[atom.btype]]

  # make type aliases for atom register arrays
  d_regs = make_mma_atom_reg_array_type(
      atom.cdtype, atom.csource, cd_reg_count)
  a_regs = make_mma_atom_reg_array_type(
      atom.atype, atom.asource, a_reg_count)
  b_regs = make_mma_atom_reg_array_type(
      atom.btype, atom.bsource, b_reg_count)
  c_regs = make_mma_atom_reg_array_type(
      atom.cdtype, atom.csource, cd_reg_count)
  if atom.kind == MmaKind.sparse and atom.sm_arch in [80, 90]:
    e_operand_count = 2 # sp-meta (reg), sp-sel (const)
    e_regs = make_mma_atom_reg_array_type(DataType.u32, MmaOperandSource.reg, 1)
  else:
    e_operand_count = 0

  # Format all strings based on computed stats
  atom_header = generic_mma_atom_format(mma_atom_header_template, atom)
  mma_atom_name = make_mma_atom_name(atom, mma_atom_name_template)
  fma_exploded_args = make_mma_atom_fma_exploded_args(
      atom, cd_reg_count, a_reg_count, b_reg_count, fma_args_indent)
  ptx_template = get_ptx_template(atom)
  ptx_instruction = make_mma_atom_instruction(ptx_template, atom)

  # ptx instruction operands
  d_start = 0
  d_stop = cd_reg_count
  a_start = d_stop
  a_stop = a_start + a_reg_count
  b_start = a_stop
  b_stop = b_start + b_reg_count
  c_start = b_stop
  c_stop = c_start + cd_reg_count
  # skip C argument in GMMA
  e_start = c_start if (atom.sm_arch == 90 and atom.threads == 128) else c_stop
  e_stop = e_start + e_operand_count

  # "%, " accounts for the other charecters
  max_operand_width = len("%, " + str(c_stop + 1))
  d_ptx_operands = make_ptx_operands(
      d_start, d_stop, max_operand_width, 8, ',', ptx_indent)

  if atom.asource == MmaOperandSource.smem_desc:
    a_ptx_operands = "\" %{},\"".format(a_start)
    synclog_a_tag = "smem"
    synclog_a_arg = ", desc_a"
  else:
    a_ptx_operands = make_ptx_operands(
        a_start, a_stop, max_operand_width, 8, ',', ptx_indent)
    synclog_a_tag = "reg"
    synclog_a_arg = ""

  if atom.bsource == MmaOperandSource.smem_desc:
    b_ptx_operands = "\" %{},\"".format(b_start)
    synclog_b_tag = "smem"
    synclog_b_arg = ", desc_b"
  else:
    b_ptx_operands = make_ptx_operands(
        b_start, b_stop, max_operand_width, 8, ',', ptx_indent)
    synclog_b_tag = "reg"
    synclog_b_arg = ""
    
  if atom.kind == MmaKind.sparse and atom.sm_arch in [80, 90]:
    e_ptx_operands = "\" %{}, %{},\"".format(e_start, e_start+1)

  instruction_end = ";\\n"
  if (atom.sm_arch == 90 and atom.threads == 128):
    mod_ptx_operands = make_wgmma_mod_ptx_operands(
        atom, e_stop, max_operand_width, 8, instruction_end, ptx_indent)
    scaleD_ptx_operand = "%{}".format(e_stop)
  elif sum(atom.blockscale_vecsize) > 0:
    c_ptx_operands = make_ptx_operands(
        c_start, c_stop, max_operand_width, 8, ',', ptx_indent)
  else:
    c_ptx_operands = make_ptx_operands(
        c_start, c_stop, max_operand_width, 8, instruction_end, ptx_indent)

  ptx_bindings = make_ptx_bindings(
      atom, cd_reg_count, a_reg_count, b_reg_count, ptx_indent)

  # Strip off the cutlass:: namespace so that the CuTe alias is used
  a_type = DataTypeTag[atom.atype].split("cutlass::")[-1]
  b_type = DataTypeTag[atom.btype].split("cutlass::")[-1]
  c_type = DataTypeTag[atom.cdtype].split("cutlass::")[-1]
 
  if sum(atom.blockscale_vecsize) > 0:
    sfa_regs = make_mma_atom_reg_array_type(
        DataType.u8, MmaOperandSource.reg, 1)
    sfb_regs = make_mma_atom_reg_array_type(
        DataType.u8, MmaOperandSource.reg, 1)

    sfa_start = c_stop
    sfa_stop = sfa_start + 1

    sfa_index_start = sfa_stop
    sfa_index_stop = sfa_index_start + 2

    sfb_start = sfa_index_stop
    sfb_stop = sfb_start + 1

    sfb_index_start = sfb_stop
    sfb_index_stop = sfb_index_start + 2

    sf_type = DataTypeTag[atom.sftype].split("cutlass::")[-1]
    sfa_ptx_operands = make_ptx_operands(
        sfa_start, sfa_stop, max_operand_width, 8, ',', ptx_indent)
    sfb_ptx_operands = make_ptx_operands(
        sfb_start, sfb_stop, max_operand_width, 8, ',', ptx_indent)

    sfa_index_ptx_operands = make_ptx_operands(
        sfa_index_start, sfa_index_stop, max_operand_width, 8, ',', ptx_indent)
    sfb_index_ptx_operands = make_ptx_operands(
        sfb_index_start, sfb_index_stop, max_operand_width, 8, instruction_end, ptx_indent)

    ptx_instruction_vs32 = make_blockscaled_mma_atom_instruction(ptx_template, atom, 32)
    ptx_instruction_vs16 = make_blockscaled_mma_atom_instruction(ptx_template, atom, 16)

  file.write(atom_header)
  file.write('\n')
  file.write(mma_atom_wrapper_template.format(**locals()))
