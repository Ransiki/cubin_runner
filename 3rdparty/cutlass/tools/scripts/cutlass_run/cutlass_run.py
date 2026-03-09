#!/usr/bin/env python

from helpers.sys_config      import augment_lib, print_lib, redirect_glibc_backtraces, get_cuda_device_ids, set_gpu_power_profiles
from helpers.label_parser    import get_labels_from_file
from helpers.layer_parser    import gen_layers_from_file
from helpers.utility         import split_comma, flags_from_descs_str, cutlass_flags_from_descs_str, prRed, prGreen, run_shell_command
from helpers.cutlass_interface import RunCache, generate_cutlass_profiler_flags, overwrite_cutlass_profiler_flags_with_cmd_line
from helpers.output_logger   import get_gpu_power_profiles

import helpers.output_logger
from datetime import datetime

from collections             import deque
import argparse
import sys, os, pdb, re

#*******************************************************************************
#* Argument parsing logic
#*******************************************************************************
def make_help(s, has_choices=False):
    result = s + " [default: %(default)s]"

    if(has_choices):
        result += " [choices: %(choices)s]"

    return result

def cutlass_run_argparser():
    '''Parse command line argument for cudnn_run.py'''
    arg_format = lambda prog: argparse.HelpFormatter(prog,max_help_position=100, width=100)

    parser = argparse.ArgumentParser(description='cutlass profiler run scripts', formatter_class=arg_format)

    parser._optionals.title = "Help Options"

    # General Options
    gen_args   = parser.add_argument_group('General Options')

    gen_args.add_argument('-dryrun', '--dryrun', '-n',
            '-n',
            action='store_const',
            const=True,
            default=False,
            help=make_help("Only print test names; do not execute"))

    gen_args.add_argument('-emit_conv2d_layers', '--emit-conv2d-layers',
            action='store_const',
            const=True,
            default=False,
            help=make_help("Emits conv2d layers for the profiler"))

    gen_args.add_argument('-testsList_batch_size', '--tests-list-batch-size', 
            metavar='N', 
            dest='testsList_batch_size', 
            default=1, 
            help=make_help("The number of tests to write together using -testsList (1 disables this)"))

    gen_args.add_argument('-global_flags', '--global-flags', '-g',
            metavar='"str"',
            dest='global_flags_str',
            default=None,
            help=make_help("Specify flags to set for all layers (example: \"-overrides algo:0,1,2 | P: h, s\""))

    gen_args.add_argument('-config', '--config', '-C',
            metavar='"str"',
            dest='filter_config_str',
            default=None,
            help=make_help("Specify filters defining current system/test configuration"))

    gen_args.add_argument('-cutlass_flags', '--cutlass-flags',
            metavar='"str"',
            dest='cutlass_flags_str',
            default=None,
            help=make_help("Specify flags to set for all layers (example: A:f16\"\""))

    gen_args.add_argument('-dump_to_log', metavar='"str"', dest='dump_to_log_str', default=None, help=make_help("Specify log file"))

    gen_args.add_argument('-device', '--device', '-d',
            metavar='n',
            type=int,
            dest='device',
            default=-1,
            help=make_help("Specify device index ('-d' flag)"))

    gen_args.add_argument('-sweep_heurgen', action='store_const', const=True, default=False, help=make_help("Use heurgen backdoor to sweep across all possible values"))

    gen_args.add_argument('-partition',
            metavar="partIndex,partCount",
            dest='partition_str',
            default=None,
            help=make_help("Partition layers and run only one partition (example: \"4,10\" will run partition #4 of 10 partitions of layers)"))

    gen_args.add_argument('-API_log_test', action='store_const', const=True, default=False, help=make_help("Enable UID comparison for API logging and test command generation, should run without randomization"))

    gen_args.add_argument('-pre_flags', '--pre-flags',
            metavar = 'PREFLAGS',
            dest    = 'pre_flags_str',
            default = "",
            help    = make_help("Specify pre_flags to place before cudnnTest"))

    # Whitelist arguments
    white_args = parser.add_argument_group('Whitelist Options')
    white_args.add_argument('-whitelist_flags', '--whitelist-flags',
            metavar = '"str"',
            dest    = 'whitelist_flags_str',
            default = None,
            help    = make_help("Only allow cases within the given flags (example: \"-whitelist_flags R: conv\""))

    white_args.add_argument('-whitelist_layer_name', '--whitelist-layer-name',
            metavar = '"str"',
            dest    = 'whitelist_layer_name',
            default = ".*",
            help    = make_help("Only allow cases where layer name matches the given regex"))

    # Caching arguments
    cache_args = parser.add_argument_group('Caching Options')

    cache_args.add_argument('-cache_path', metavar='"str"', dest='cache_path', default=None, help=make_help("Specify cache path; default assumes no caching. A cache stores results of calls so that another run of cudnn_perf.py can reload the results."))

    cache_args.add_argument('-cache_freq', metavar='N', dest='cache_freq', default=-1, help=make_help("Specify interval to update cache; 2 means update after every other test call. The special case of -1 will only update cache after all calls are finished."))

    # Path arguments
    path_args  = parser.add_argument_group('Path Options')

    path_args.add_argument('-binpath', '--bin-path',
            metavar='"path"',
            dest='bin_path',
            default='./../../../bin/x86_64_Linux_release',
            help=make_help("Specify bin path (where binary is located)"))

    path_args.add_argument('-bin_name', '--bin-name',
            metavar='"str"',
            dest='bin_name',
            default='cutlass_profiler',
            choices=["cutlass_profiler"],
            help=make_help("Specify which binary to run", True))

    path_args.add_argument('-libpath', '--lib-path', '-L',
            metavar='"path"',
            dest='lib_path',
            default='./../../../bin/x86_64_Linux_release',
            help=make_help("Specify lib path (where libcudnn.so is located)"))


    path_args.add_argument('--use-dir',
            action  = 'store_true',
            default = False,
            help    = 'whether to use the new labels/layers dirs hotness')

    path_args.add_argument('-layer_file', '--layer-file',
            metavar = '"path"',
            dest    = 'layers_path',
            default = 'none.layer',
            help    = make_help("Specify layers file (path to layer_definitions"))

    path_args.add_argument('-label_file', '--label-file',
            metavar = '"path"',
            dest    = 'labels_path',
            default = 'none.label',
            help    = make_help("Specify labels file (path to layer_labels), can be \"None\" if no label is needed"))

    path_args.add_argument('-csv_filename', '--csv-filename',
            metavar = '"path"',
            dest    = 'csv_filename',
            default = 'default.csv',
            help    = make_help("Specify file name for csv output"))

    path_args.add_argument('-csv_path', '--csv-path',
            metavar = '"path"',
            dest    = 'csv_path',
            default = './logs_csv/',
            help    = make_help("Specify directory to generate csv logs. If the directory is not present, it'll be created"))

    path_args.add_argument('-permit_unlocked_clocks', '--permit-unlocked-clocks',
            action  = 'store_const',
            const   = True,
            dest    = 'permit_unlocked_clocks',
            default = False,
            help    = make_help("When enabled, the script will not set any gpu clocks"))

    return parser

#
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

def emit_conv2d_network_py(layers):

  layer_template = """
  ${network}.add_layer(Layer(
      [${n}, ${h}, ${w}, ${c}],
      [${k}, ${r}, ${s}, ${c}],
      [${pad_h}, 0, ${pad_w}, 0],
      [${stride_h}, ${stride_w}],
      [${dilation_h}, ${dilation_w}]    
  ));
  """

  for layer in layers:
    values = {
      'network': layer.test_name.split('_')[0],
      'n': '1',
      'h': layer.flags['input-size::h'][0],
      'w': layer.flags['input-size::w'][0],
      'c': layer.flags['input-size::c'][0],
      'k': layer.flags['filter-size::k'][0],
      'r': layer.flags['filter-size::r'][0],
      's': layer.flags['filter-size::s'][0],
      'pad_h': layer.flags['pad::top'][0],
      'pad_w': layer.flags['pad::left'][0],
      'stride_h': layer.flags['stride::h'][0],
      'stride_w': layer.flags['stride::w'][0],
      'dilation_h': layer.flags['dilation::h'][0] if 'dilation::h' in layer.flags else '0',
      'dilation_w': layer.flags['dilation::w'][0] if 'dilation::w' in layer.flags else '0', 
    }

    print(SubstituteTemplate(layer_template, values))
    



def post_parse_args(parsedargs):
    '''Make some reasonble check of parsed arguments and throw exception if argument cannot be interpreted.'''
    layers_dir = labels_dir = scripts_dir = os.path.dirname(__file__) # scripts_dir is where cudnn_run.py resides
    if parsedargs.use_dir:
        # Assume layer and label files are stored in scripts_dir/layers and scripts_dir/labels subdirectories, respectively
        layers_dir = os.path.join(scripts_dir, 'layers')
        labels_dir = os.path.join(scripts_dir, 'labels')
    # Check --layer-file argument
    if os.path.basename(parsedargs.layers_path) == parsedargs.layers_path:  # only basename provided through argument
        test_path = os.path.join(layers_dir, parsedargs.layers_path)
        if os.path.isfile(test_path):
            parsedargs.layers_path = test_path
    if not os.path.isfile(parsedargs.layers_path): # assert that the layer file exists
        raise IOError(u'layer file {} is not a valid path'.format(parsedargs.layers_path))

    # Check --label-file argument
    if os.path.basename(parsedargs.labels_path) == parsedargs.labels_path:  # only basename provided
        test_path = os.path.join(labels_dir, parsedargs.labels_path)
        if os.path.isfile(test_path):
            parsedargs.labels_path = test_path
    if not os.path.isfile(parsedargs.labels_path): # assert that the label file exists
        raise IOError(u'label file {} is not a valid path'.format(parsedargs.labels_path))

    if parsedargs.sweep_heurgen and parsedargs.dryrun:
        raise ValueError("[INVALID FLAGS] Unable to run sweep_heurgen with dryrun")

    if(parsedargs.dump_to_log_str):
        sys.stdout = Logger(parsedargs.dump_to_log_str)

    # Print all parsed arguments
    print "Printing all command line arguments"
    for arg in parsedargs.__dict__:
        print ("\t%s = %s" % (arg, parsedargs.__dict__[arg]))

    return parsedargs

parsed_args = post_parse_args(cutlass_run_argparser().parse_args())

#*******************************************************************************
#* Setup LD_LIBRARY_PATH (os-agnostic)
#*******************************************************************************
# Add lib_path to OS-dependant library path
if(parsed_args.lib_path != ''):
    augment_lib(parsed_args.lib_path)

# Print current library info
print "\n\nPrinting current LD_LIBRARY_PATH path"
print_lib()
print ""

# Redict glibc backtrace to retrieve errors from test calls
redirect_glibc_backtraces()


#*******************************************************************************
#* Initialize all default values
#******************************************************************************/
# Track results of all tests
counts = {"total": 0.0, "passed": 0.0, "waived": 0.0, "failed": 0.0}

# Global filters (list of filters defined by user)
filter_config = split_comma(parsed_args.filter_config_str)


#*******************************************************************************
#* Initialize all default values
#******************************************************************************/
# Track results of all tests
counts = {"total": 0.0, "passed": 0.0, "failed": 0.0,  "unknown": 0.0}


#*******************************************************************************
#* Layer Generation & Filtration
#*******************************************************************************
# Add filter for GPU speed. This can be used to filter long running tests on slower GPUs

# Hold test results (to handle duplicates efficiently)
results_cache = {}

# Store all layers that aren't filtered out
filtered_layers = []

#*******************************************************************************
#* Layer Generation & Filtration
#*******************************************************************************
# Get labels
if parsed_args.labels_path.lower() != "none":
    labels = get_labels_from_file(parsed_args.labels_path, filter_config)
else:
    labels = None

global_device_flag = "d: %d" % parsed_args.device

if parsed_args.global_flags_str != None:
    global_flags_str = parsed_args.global_flags_str + " * " + global_device_flag
else:
    global_flags_str = global_device_flag

# Global flags (will take priority over layer/label definitions)
global_flags_list = flags_from_descs_str(global_flags_str, labels)

# Flag Whitelist
whitelist_flags_list = flags_from_descs_str(parsed_args.whitelist_flags_str, labels)

# Get layers
layers = gen_layers_from_file(parsed_args.layers_path, whitelist_flags_list, parsed_args.whitelist_layer_name, global_flags_list, labels)

#*******************************************************************************
#* Cache initialization
#*******************************************************************************
cache = RunCache(batch_size = parsed_args.testsList_batch_size, max_cache = max(100000, int(parsed_args.testsList_batch_size)))

# cutlass specific flags from the command line
cutlass_cmd_line_flags_list = cutlass_flags_from_descs_str(parsed_args.cutlass_flags_str, labels)

# Stacking all layers in a single batch
current_batch = []
current_batch_count = 0
current_batch_map = {} # to remove duplicates

# Create log_csv directory
dir_name = parsed_args.csv_path
mkdir_shell = ["mkdir", "-p", dir_name]
run_shell_command(mkdir_shell)

cuda_device_ids = get_cuda_device_ids(parsed_args.bin_path, parsed_args.bin_name)

requested_device = parsed_args.device

if requested_device == -1:
  list_devices = cuda_device_ids.keys()
else:
  if cuda_device_ids.keys().index(str(requested_device)) == -1:
    print "Unable to find device: ", str(requested_device), " exiting.."
    exit()
  else:
    list_devices = [str(requested_device)]

type_map = {'conv' : 'conv', 'dgrad' : 'conv', 'wgrad' : 'conv', 'bgrad' : 'conv', 'gemm' : 'gemm'}

gpu_power_profiles = get_gpu_power_profiles()
for device in list_devices:
  if parsed_args.permit_unlocked_clocks == False:
    set_gpu_power_profiles(device, cuda_device_ids, gpu_power_profiles)
  layer_type = ''
  while True:
    try:
        layer = next(layers)
        layer_type = layer.flags.flags_dict['R'][0]
        cutlass_flags = generate_cutlass_profiler_flags(layer.flags, device)

        # add cutlass specific flags from command line to cutlass_flags 
        cutlass_flags = overwrite_cutlass_profiler_flags_with_cmd_line(cutlass_flags, cutlass_cmd_line_flags_list)
        layer = layer._replace(flags=cutlass_flags)
        
        # add layer to the current batch for launch via cutlass_perf_test
        current_batch_map[str(layer.flags)] = 1 # for removing duplicates
        current_batch.append(layer)
        current_batch_count += 1        
        if parsed_args.dryrun:
            print "** cuBlas/cuDNN ** commandline: %s (Layer Name: %s)" % (layer.flags.get_str(), layer.base_name)
            print "Layer Name: %s" %layer.base_name             
            print "** cutlass_profiler ** commandline: %s" %cutlass_flags
    except StopIteration:
        break

  csv_output = helpers.output_logger.log_csv(parsed_args.csv_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + parsed_args.layers_path.split('/')[-1].split('.layer')[0] + "_" + parsed_args.labels_path.split('/')[-1].split('.label')[0], type_map[layer_type])

#*******************************************************************************
#* Emit Network .py file for cutlass profiler
#*******************************************************************************
  if parsed_args.emit_conv2d_layers:
    emit_conv2d_network_py(current_batch)
    continue

#*******************************************************************************
#* Test Execution
#*******************************************************************************
  for layer in current_batch:
 
    test_name_str = "%s %s" % (parsed_args.bin_name, str(layer.flags))

    print "[CUTLASS RUN] START test %s : '%s/%s %s'" % (layer.test_name, parsed_args.bin_path, parsed_args.bin_name, str(layer.flags))

    if parsed_args.dryrun:
        print ""
        continue

    # Obtain run from cache; will run test if not available in cache
    (output, error_msg, return_code), shell_cmd = cache.run_cutlass_profiler(layer, parsed_args.bin_path, parsed_args.bin_name, pre_flags_str=parsed_args.pre_flags_str)

    csv_output.add_to_log(layer.base_name, shell_cmd, output, return_code)

    # Use return code to determine PASS/FAIL/WAIVE
    if(error_msg == None):
        print "[CUTLASS RUN] END test %s %s" % (prGreen("PASSED"), (test_name_str))
        counts["passed"] += 1
        current_test_status = "PASSED"        
        # for the passed cutlass_perf_test run display output inside cutlass_interface::process_n_display_each_kernel_run
    else:
        print "[CUTLASS RUN] END test %s %s" % (prRed("FAILED"), (test_name_str))
        counts["failed"] += 1
        current_test_status = "FAILED"
        # for the failed cutlass_perf_test run display output here
        # print output
        print "[CUTLASS RUN] Error Detected: %s" % error_msg
    print ""
    counts["total"] += 1

  sys.stdout.flush()

  csv_output.write_to_csv()


#*******************************************************************************
#* Print summaries of results
#******************************************************************************/
  print ""
  print "*** LAYER TEST RESULT  ***"
  print "Total tests   : %d" % counts["total"]
  print "Failures      : %d" % counts["failed"]
  print "Successes     : %d" % counts["passed"]
  if (counts["passed"] + counts["failed"]):
    print "Basic Sanity  : %4.2f%%" % ((100*counts["passed"]) / (counts["passed"] + counts["failed"]))
  print "Total tests number is the number of cutlass_profiler commands launched. Each cutlass_profiler command may launch multiple kernels.\n"
  sys.stdout.flush()
