import enum, re, pdb
import os.path
import shutil
import argparse

# cutlass conv3d problem size ctor template for unit test
conv3d_problem_size_unit_test_template = """
  conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
    {batch_size, ${d}, ${h}, ${w}, ${c}},     // input size  (NDHWC)
    {${k}, ${t}, ${r}, ${s}, ${c}},              // filter size (KTRSC)
    cutlass::Coord<3>({${pad_d}, ${pad_h}, ${pad_w}}),    // padding (pad_d, pad_h, pad_w)
    cutlass::Coord<3>({${stride_d}, ${stride_h}, ${stride_w}}),    // stride (stride_d, stride_h, stride_w)
    cutlass::Coord<3>({${dilation_d}, ${dilation_h}, ${dilation_w}})     // dilation (dilation_d, dilation_h, dilation_w) 
  ));
"""

# cutlass conv3d problem size ctor template for perf test
conv3d_problem_size_perf_test_template = """
  LayerName.add_layer(Layer3D(
    [${n}, ${d}, ${h}, ${w}, ${c}], \\
    [${k}, ${t}, ${r}, ${s}, ${c}], \\
    [${pad_d}, ${pad_h}, ${pad_w}], \\
    [${stride_d}, ${stride_h}, ${stride_w}], \\
    [${dilation_d}, ${dilation_h}, ${dilation_w}], \\
  ));
"""

# substitue template with values
def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

# define layer patterns
layer_name_pattern=re.compile('"(.*)" = .*')
ncdhw_pattern=re.compile('dimA:"(\d+),(\d+),(\d+),(\d+),(\d+)"')
kctrs_pattern=re.compile('filtA:"(\d+),(\d+),(\d+),(\d+),(\d+)"')
padding_pattern=re.compile('padA:"(\d+),(\d+),(\d+)"')
stride_pattern=re.compile('convStrideA:"(\d+),(\d+),(\d+)"')
dilation_pattern=re.compile('dilationA:"(\d+),(\d+),(\d+)"')

# generic pattern extractor
def get_pattern(layer, pattern):
  match = re.search(pattern, layer)
  if match:
    return match.groups()
  else:
    return False

# function to emit cutlass conv3d problem size ctor
def emit_conv3d_problem_sizes(args):

  layer_file = open(args.layer_file, 'r')
  lines = layer_file.readlines()

  for line in lines:
    layer = {}

    # skip commented layers
    if line.startswith("#"):
      continue

    layer['name'] = get_pattern(line, layer_name_pattern)[0]
    if args.layer_name != None and args.layer_name not in layer['name']:
      continue

    # emit layer description
    #print 'Emitting layer:' + line
    ncdhw = get_pattern(line, ncdhw_pattern)
    layer['n'] = ncdhw[0]
    layer['c'] = ncdhw[1]
    layer['d'] = ncdhw[2]
    layer['h'] = ncdhw[3]
    layer['w'] = ncdhw[4]


    kctrs = get_pattern(line, kctrs_pattern)
    layer['k'] = kctrs[0]
    layer['c'] = kctrs[1]
    layer['t'] = kctrs[2]
    layer['r'] = kctrs[3]
    layer['s'] = kctrs[4]

    padding = get_pattern(line, padding_pattern)
    layer['pad_d'] = padding[0]
    layer['pad_h'] = padding[1]
    layer['pad_w'] = padding[2]

    stride = get_pattern(line, stride_pattern)
    layer['stride_d'] = stride[0]
    layer['stride_h'] = stride[1]
    layer['stride_w'] = stride[2]

    dilation = get_pattern(line, dilation_pattern)
    layer['dilation_d'] = dilation[0]
    layer['dilation_h'] = dilation[1]
    layer['dilation_w'] = dilation[2]

    problem_template = conv3d_problem_size_unit_test_template

    if args.emit_problem_size_for == 'perf-test':
      problem_template = conv3d_problem_size_perf_test_template

    print(SubstituteTemplate(problem_template, layer))

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Parses conv3d layers and emits cutlass conv3d problem sizes")
  parser.add_argument('--layer-file', type=str, default=None, required=True, help='Full path of layer file')
  parser.add_argument('--layer-name', type=str, default=None, required=False, help='Layer name to extract')
  parser.add_argument('--emit-problem-size-for', type=str, default='unit-test', required=False, help='Emit ctor for (unit-test, perf-test)')

  args = parser.parse_args()

  emit_conv3d_problem_sizes(args)