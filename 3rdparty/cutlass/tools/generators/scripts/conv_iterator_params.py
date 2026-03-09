
import os
import sys
import csv

def EmitDeclarations():
  return [
    '',
    '#include <unordered_map>',
    '#include <algorithm>',
    '',
    'enum class OpcodeClass {',
    '  kSimt,',
    '  kTensorOp',
    '};',
    '',
    'struct Conv2dTile {',
    '  OpcodeClass opcode_class;',
    '  int element_size;',
    '  int cta_rows;',
    '  int cta_columns;',
    '  int threads;',
    '  int access_size;',
    '',
    '  bool operator==(Conv2dTile const &rhs) const {',
    '    return (opcode_class == rhs.opcode_class) &&',
    '      (element_size == rhs.element_size) &&',
    '      (cta_rows  == rhs.cta_rows) &&',
    '      (cta_columns == rhs.cta_columns) &&',
    '      (threads == rhs.threads) &&',
    '      (access_size == rhs.access_size);',
    '  }',
    '};',
    '',
    'struct Conv2dIteratorParams {',
    '  int iterations_contiguous;',
    '  int iterations_strided;',
    '  int delta_contiguous;',
    '  int delta_strided;',
    '};',
    '',
    'cutlass::layout::PitchLinearCoord ThreadmapIterations(Conv2dIteratorParams const &params) {',
    '  return cutlass::layout::PitchLinearCoord(params.iterations_contiguous, params.iterations_strided);',
    '}',
    '',
    'layout::PitchLinearCoord ThreadmapDelta(Conv2dIteratorParams const &params) {',
    '  return cutlass::layout::PitchLinearCoord(params.delta_contiguous, params.delta_strided);',
    '}',
    '',
    'struct Conv2dTileHasher {',
    '  using IntHash = std::hash<int>;',
    '',
    '  inline static size_t rotl(size_t key, int shl) {',
    '    return (key << shl) | (key >> (sizeof(key)*8 - shl));',
    '  }',
    '',
    '  inline size_t operator()(Conv2dTile const &key) const {',
    '    IntHash hash;',
    '',
    '    return ',
    '      rotl(hash(int(key.opcode_class)), 1) ^ ',
    '      rotl(hash(int(key.element_size)), 2) ^ ',
    '      rotl(hash(int(key.cta_rows)), 3) ^ ',
    '      rotl(hash(int(key.cta_columns)), 4) ^',
    '      rotl(hash(int(key.threads)), 5) ^',
    '      rotl(hash(int(key.access_size)), 6);',
    '  }',
    '};',
    '',
    'using Conv2dTileMap = std::unordered_map<Conv2dTile, Conv2dIteratorParams, Conv2dTileHasher>;',
    '',
    ''
  ]

def EmitMapDefinitionBegin(conv_operator, operand):
  return [
    'Conv2dTileMap %s_%s_invariants = {' % (conv_operator, operand),
  ]

def EmitMapDefinitionEnd(conv_operator, operand):
  return ['};',]


def Main():

  conv_operators = [
    ('conv2d_fprop', ['activation', 'filter']),
    ('conv2d_dgrad', ['output_gradient', 'filter']),
    ('conv2d_wgrad', ['output_gradient', 'activation']),
  ]

  lines = EmitDeclarations()

  for conv_operator, operands in conv_operators:
    for operand in operands:

      lines += EmitMapDefinitionBegin(conv_operator, operand)

      initializers = []

      definitions = {}

      with open('conv_iterator_params.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
          if row['Operator'] == conv_operator and row['Operand'] == operand:
            items = (
              int(row['ElementSize']),
              int(row['CtaRows']),
              int(row['CtaColumns']),
              int(row['ThreadCount']),
              int(row['AccessSize']),
              int(row['IterationsContiguous']),
              int(row['IterationsStrided']),
              int(row['DeltaContiguous']),
              int(row['DeltaStrided']),
            )
            key = (
              int(row['ElementSize']),
              int(row['CtaRows']),
              int(row['CtaColumns']),
              int(row['ThreadCount']),
              int(row['AccessSize']),
            )
            value = (
              int(row['IterationsContiguous']),
              int(row['IterationsStrided']),
              int(row['DeltaContiguous']),
              int(row['DeltaStrided']),
            )
            if key not in definitions.keys():
              initializers.append('  { {OpcodeClass::kTensorOp, %d, %d, %d, %d, %d}, {%d, %d, %d, %d} }' % items)
              definitions[key] = value
            elif value != definitions[key]:
              print("#error Collision on key", str(key))

      lines.append(',\n'.join(initializers))

      lines += EmitMapDefinitionEnd(conv_operator, operand)
      lines.append('')

  print('\n'.join(lines))
  return 0

if __name__ == '__main__':
  sys.exit(Main())
