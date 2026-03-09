#  module serves as an interface between Python and cudnnTest/cublasTest

# To accomplish this, it runs cudnnTest and parses the output to obtain result info.
#   - Result info includes elapsed time, M, N, K, algo, etc...
#   - See "def parse(output)" for a definition of all parsed data.

# The main use of this module is the function "run_flags(flags)"
#   - This will run cutlass_profiler and return parsed output


import re           # For regular expressions
import subprocess   # To spawn processes (cudnnTest)
import collections  # For named tuple
import os           # To check if file exists

from utility     import split_space, flags_from_descs_str, flags_match_a_in_b
from Flags       import Flags
import pdb

def generate_cutlass_profiler_flags(layer_flags, device):

  # Layer cutlass string mappings
  translate_to_cutlass_datatype = {'h':'f16', 's':'f32', 'd':'f64', 'nc32':'s32', 'b':'s8'}

  # Flags(cutlass_flag=True, None, None) creates a cutlass flag
  cutlass_flags = Flags(True, None, None)

  '''
  # Set device for cutlass_profiler
  if 'd' in layer_flags.flags_dict:
    cutlass_flags['device'] = (layer_flags['d'][0],)
  '''

  cutlass_flags['device'] = (device,)

  ## START ## setting FunctionSchema cutlass_profiler cmd line for Gemm ## START ###
  if  layer_flags.flags_dict['R'][0] == 'gemm':
    cutlass_flags['function'] = ('Gemm',)
    
    # Set gemm problem-size
    if ('m' in layer_flags.flags_dict):
      cutlass_flags['problem-size::m'] = (layer_flags['m'][0],)
    if ('n' in layer_flags.flags_dict):
      cutlass_flags['problem-size::n'] = (layer_flags['n'][0],)
    if ('k' in layer_flags.flags_dict):
      cutlass_flags['problem-size::k'] = (layer_flags['k'][0],)        
    
    # input tensors (TensorA and TensorB) datatype and laypout
    if ('Pin' in layer_flags.flags_dict):
      if(('ta' not in layer_flags.flags_dict) and ('tb' not in layer_flags.flags_dict)):
        cutlass_flags['A'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":col",)
        cutlass_flags['B'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":col",)
      if(('ta' in layer_flags.flags_dict) and ('tb' not in layer_flags.flags_dict)):
        cutlass_flags['A'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":row",)
        cutlass_flags['B'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":col",)
      if(('ta' not in layer_flags.flags_dict) and ('tb' in layer_flags.flags_dict)):
        cutlass_flags['A'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":col",)
        cutlass_flags['B'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":row",)
      if(('ta' in layer_flags.flags_dict) and ('tb' in layer_flags.flags_dict)):
        cutlass_flags['A'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":row",)
        cutlass_flags['B'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":row",)
      if (('Pout' in layer_flags.flags_dict)):
        cutlass_flags['C'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pout'][0]],)
      if (('Pcomp' in layer_flags.flags_dict)):
        cutlass_flags['accumulator-type'] = (translate_to_cutlass_datatype[layer_flags.flags_dict['Pcomp'][0]],)

    # Set alpha and beta
    if ('A' in layer_flags.flags_dict):
      cutlass_flags['alpha'] =  (layer_flags.flags_dict['A'],)

    if ('B' in layer_flags.flags_dict):
      cutlass_flags['beta'] =  (layer_flags.flags_dict['B'],)
  ## END ## setting FunctionSchema cutlass_profiler cmd line for Gemm ## END ###

  ## START ## setting FunctionSchema cutlass_profiler cmd line for Conv2dFprop ## START ###
  elif  (layer_flags.flags_dict['R'][0] == 'conv') or (layer_flags.flags_dict['R'][0] == 'dgrad') or (layer_flags.flags_dict['R'][0] == 'wgrad'):

    conv_cudnn_cutlass_op_map = {'conv':'Conv2dFprop', 'dgrad':'Conv2dDgrad', 'wgrad':'Conv2dWgrad'}

    cutlass_flags['function'] = (conv_cudnn_cutlass_op_map[layer_flags.flags_dict['R'][0]],)

    # conv2d input-size
    if ('dimA' in layer_flags.flags_dict):
      extent = layer_flags['dimA'][0].split(',')
      cutlass_flags['input-size::n'] = (extent[0],)
      # overwrite input batch-size if -n in layer_flags
      if ('n' in layer_flags.flags_dict):
        cutlass_flags['input-size::n'] = (layer_flags['n'][0],)
      cutlass_flags['input-size::c'] = (extent[1],)
      cutlass_flags['input-size::h'] = (extent[2],)
      cutlass_flags['input-size::w'] = (extent[3],)


    # conv2d filter-size
    if ('filtA' in layer_flags.flags_dict):
      extent = layer_flags['filtA'][0].split(',')
      cutlass_flags['filter-size::k'] = (extent[0],)
      cutlass_flags['filter-size::c'] = (extent[1],)
      cutlass_flags['filter-size::r'] = (extent[2],)
      cutlass_flags['filter-size::s'] = (extent[3],) 

    # input tensors (Activation, Filter) datatype and layout. TODO: For now, only packed_nhwc is supported by cutlass div lib
    if ('Pin' in layer_flags.flags_dict):
      cutlass_flags['Activation'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":packed_nhwc",)
      cutlass_flags['Filter'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pin'][0]]+":packed_nhwc",)

    # output tensor (Output) datatype and layout. TODO: For now, only packed_nhwc is supported by cutlass div lib
    if ('Pout' in layer_flags.flags_dict):
      cutlass_flags['Output'] =  (translate_to_cutlass_datatype[layer_flags.flags_dict['Pout'][0]]+":packed_nhwc",)

    # convolution mode (crosscorrelation or convolution)
    if ('x' in layer_flags):
      cutlass_flags['conv-mode'] =  ('crosscorrelation',)

    # padding
    if (('pad_h' in layer_flags) or ('pad_w' in layer_flags)):
      if(layer_flags['pad_h'][0] != layer_flags['pad_w'][0]):
        raise Exception("[CUTLASS RUN SCRIPT FAILED] asymmetric padding not supported pad_h != pad_w")
      cutlass_flags['pad::top'] =  (layer_flags['pad_h'][0],)
      cutlass_flags['pad::bottom'] =  (layer_flags['pad_h'][0],)
      cutlass_flags['pad::left'] =  (layer_flags['pad_h'][0],)
      cutlass_flags['pad::right'] =  (layer_flags['pad_h'][0],)

    if ('padA' in layer_flags):
      padding = layer_flags['padA'][0].split(',')
      cutlass_flags['pad::top'] =  (padding[0],)
      cutlass_flags['pad::bottom'] =  (padding[0],)
      cutlass_flags['pad::left'] =  (padding[0],)
      cutlass_flags['pad::right'] =  (padding[0],)

    # stride
    if ('u' in layer_flags):
      cutlass_flags['stride::h'] =  (layer_flags['u'][0],)
    if ('v' in layer_flags):
      cutlass_flags['stride::w'] =  (layer_flags['v'][0],)

    # dilation
    if ('dilation_h' in layer_flags):
      cutlass_flags['dilation::h'] =  (layer_flags['dilation_h'][0],)
    if ('dilation_w' in layer_flags):
      cutlass_flags['dilation::w'] =  (layer_flags['dilation_w'][0],)
  ## END ## setting FunctionSchema cutlass_profiler cmd line for Conv2dFprop ## END ###

  return cutlass_flags

# Define patterns and tuples for all parsed outputs A LogExtractor consist of a pattern for matching,
# and a namedtuple type for matched group.
# Only type with single field is extractable presently.
class LogExtractor(collections.namedtuple('_NT_LogExtractor', ('pattern', 'nt_type'))):
    '''Extractor information from output log'''
    name2extractor_map = collections.OrderedDict()
    @classmethod
    def addExtractor(cls, name, pattern, fields, re_flags=0):
        '''Add a new extractor (or replaces old one) by {name} based on {pattern},
        with namedtuple(typename, fields) output type, where typename is the string name prepended by _NT_

        {pattern} can be a format string for regular expression.  The following fields are supported:
            FLOAT_GROUP
            INT_GROUP
        '''
        typename = '_NT_{}'.format('_'.join(name.capitalize().split())) # auto-generate namedtuple type name, no space allowed.
        pattern = pattern.format( # substitute common matching group
                FLOAT_GROUP = r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
                INT_GROUP   = r'([+-]?\d+)',
                UINT_GROUP  = r'(\d+)',
                UID_GROUP   = r'([\da-z]{32})')
        inst = cls (
                pattern = re.compile(pattern, re_flags),
                nt_type = collections.namedtuple(typename, fields))
        cls.name2extractor_map[name] = inst # NOTE: updating key does not change the order

    @classmethod
    def getExtratorList(cls):
        '''Return a list of string of valid --extract arguments'''
        return [name if len(extractor.nt_type._fields) == 1  # --extract name for single field extractor
                else '{}.{}'.format(name, field)                     # --extract name.field for multiple field extractor
                for name, extractor in cls.name2extractor_map.iteritems()
                for field in extractor.nt_type._fields
                ]

    def extractToTuple(self, output):
        '''Search {output} for pattern, if found return namedtuple instance for it'''
        match = self.pattern.search(output)
        if match:
            num_fields = len(self.nt_type._fields)
            matched_groups = match.groups()
            if len(matched_groups) != num_fields:
                raise Exception("Unable to fit groups {:s} into Tuple \"{:s}\" with fields {:s}"
                        .format(matched_groups, self.nt_type.__name__, self.nt_type._fields))
            return self.nt_type(*(matched_groups[:num_fields]))
        return None

    def make_default_tuple(self, defval=None):
        '''Return an instance of self.nt_type with {defval} for all fields'''
        return self.nt_type(*((defval,) * len(self.nt_type._fields)))


# Extractors are added in order:
# Extract cutlass specific numbers

LogExtractor.addExtractor( # provider of the kernel
        name     = 'provider',        
        pattern  = r'Provider: (.*)',
        fields   = "kernel_provider",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # Schmea
        name     = 'schema',        
        pattern  = r'Schema: (.*)',
        fields   = "name",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # FunctionType
        name     = 'function_type',        
        pattern  = r'Type: (.*)',
        fields   = "name",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # Function
        name     = 'function',        
        pattern  = r'Function: (.*)',
        fields   = "name",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # Arguments
        name     = 'arguments',
        pattern  = r'Arguments: (\d+), (\d+), (\d+), (.*):(.*), (.*):(.*), (.*):(.*), (.*), (.*), (\d+), (.*), (.*), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)',
        fields   = "m n k datatype_a layout_a datatype_b layout_b datatype_c layout_c alpha beta split_k opcode_class accumulator_type cta_m cta_n cta_k warp_m warp_n warp_k")

LogExtractor.addExtractor( # cutlass kernel runtime
        name     = 'time',        
        pattern  = r'Runtime: (.*) ms',
        fields   = "milliseconds",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # cutlass kernel GFLOPs
        name     = 'math',        
        pattern  = r'Math: (.*) GFLOP/s',
        fields   = "gflops",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # cutlass kernel Memory bandwidth
        name     = 'memory',        
        pattern  = r'Memory: (.*) GB/s',
        fields   = "gbps",
        re_flags = re.IGNORECASE)

LogExtractor.addExtractor( # cutlass verfication result against cudnn
        name     = 'cudnn_verification',        
        pattern  = r'Verification.*?cuDNN: (.*)',
        fields   = "result",
        re_flags = re.IGNORECASE)

# Layer Name (only works on output parsed from cutlass_perf_run.py log)
LogExtractor.addExtractor(
        name     = 'layer_name',
        pattern  = r'Layer Name: (.*)',
        fields   = "layer_name")

# Test Flags (only works on output parsed from cutlass_perf_run.py log)
LogExtractor.addExtractor(
        name     = 'test_flags',
        pattern  = r'Test Flags: (.*)',
        fields   = "test_flags")

# Unique Flags (only works on output parsed from cutlass_perf_run.py log)
LogExtractor.addExtractor(
        name     = 'unique_flags',
        pattern  = r'Unique Flags: (.*)',
        fields   = "unique_flags")

# cuBLAS numbers from cutlass_profiler
LogExtractor.addExtractor( # cutlass kernel runtime
        name     = 'cublas_time',        
        pattern  = r'Provider: cuBLAS.*?Runtime: (.*) ms',
        fields   = "milliseconds",
        re_flags = re.DOTALL)

LogExtractor.addExtractor( # cutlass kernel GFLOPs
        name     = 'cublas_math',        
        pattern  = r'Provider: cuBLAS.*?Math: (.*) GFLOP/s',
        fields   = "gflops",
        re_flags = re.DOTALL)

LogExtractor.addExtractor( # cutlass kernel Memory bandwidth
        name     = 'cublas_memory',        
        pattern  = r'Provider: cuBLAS.*?Memory: (.*) GB/s',
        fields   = "gbps",
        re_flags = re.DOTALL)

class FileRange(collections.namedtuple("FileRange", "start end file_desc")):
    '''Nametuple for reading a range .start to .end of file descriptor .file_desc'''
    @classmethod
    def fromMatches(cls, file_desc, start_re, stop_re, max_segments=None, ret_matches=False):
        '''
        Return list of instances for segments in {file_desc} between lines that matches {start_re} and the next one that matches {stop_re}

        if {max_segments} is an integer, stop read after {max_segments} has been generated.
        If {ret_matches} is True, each instance is accompanied by start and stop re match objects.
        '''
        ret = []
        count = 0
        start_match = None # last start_re match
        while True: # use this instead of iter(file_desc) to avoid the read-ahead buffe, which breaks .tell()
            curr_pos = file_desc.tell()
            line = file_desc.readline()
            if not line: # empty line == '\n'
                break
            match = start_re.match(line)
            if match:
                start_pos = curr_pos
                start_match = match
            match = stop_re.match(line)
            if start_match is not None and match:
                end_pos = file_desc.tell()
                inst = cls(file_desc=file_desc, start=start_pos, end=end_pos)
                if ret_matches:
                    ret.append((start_match, match, inst))
                else:
                    ret.append(inst)
                count += 1
                if max_segments is not None and count >= max_segments:
                    break
                start_match = None # reset the start_match
        return ret

    def read(self):
        '''Read the range'''
        self.file_desc.seek(self.start)
        return self.file_desc.read(self.end - self.start)

# Create namedtuples for all parsed outputs
ParsedOutput = collections.namedtuple("ParsedOutput", LogExtractor.name2extractor_map.keys())

# Create namedtuple for RunResult
RunResult       = collections.namedtuple("RunResult", "flags bin_path bin_name output error_msg return_code parsed")

# Create namedtuple for RunListResult
RunListResult   = collections.namedtuple("RunListResult", "flags bin_path bin_name outputs error_msg return_code parsed")

# Created namedtuple for CachedRunResult
CachedRunResult = collections.namedtuple("CachedRunResult", "cache_count run_result")


class OutputParser(object):
    '''Parser for cudnnTest output, instance[name] extracts namedtuple for mathcing name'''
    # list of valid --extract arguments
    def __init__(self, output):
        self.output = output
        self.cache = {}

    def __getitem__(self, name):
        if name in LogExtractor.name2extractor_map:
            extractor = LogExtractor.name2extractor_map[name]
            if self.cache.get(name) is None:
                self.cache[name] = extractor.extractToTuple(self.output)
            return self.cache[name]
        raise Exception("Name %s is not parsable", name)

# Find the line strictly after lid in lines matching cutlass single test start pattern
def get_next(lid, lines):

    cutlass_single_test_start_pat = re.compile(r'========================== CUTLASS Kernel Performance Report ==========================', re.MULTILINE | re.DOTALL)
    for lidNext in range(lid+1, len(lines)):
        match = cutlass_single_test_start_pat.search(lines[lidNext])
        if match:
            return (lidNext, lidNext+1)
    return (len(lines), -1)

# cutlass_profiler single launch outputs multiple runs in a single output
# thus, here we seperate and extract each output, parse, and return in a format 
# that can be used for extensive kernel/use case analysis
def parse_cutlass_profiler_output(output, bin_path, bin_name):

    lines = output.split('\n')
    (lid_current, begin_line) = get_next(-1, lines)
    if begin_line != 3:
        print('[CUTLASS RUN SCRIPT FAILED] Critical error, expected to read line 1, got line {}'.format(begin_line))
        return ([output], [None])
    (lid_next, next_line) = get_next(lid_current, lines)

    parsed_result = []
    split_output = []
    expected_line = 0

    while lid_current < len(lines):
        expected_line += 1
        test_to_parse = lines[lid_current:lid_next]
        parsed_output = OutputParser('\n'.join(test_to_parse))

        ## TODO: Insert checks on individual outputs

        parsed_result.append(parsed_output)
        split_output.append('\n'.join(test_to_parse))
        (lid_current, begin_line) = (lid_next, next_line)
        (lid_next, next_line) = get_next(lid_current, lines)

    return (split_output, parsed_result)

# Runs given flags
def run_flags(flags, bin_path, bin_name, piped_input=None, pre_flags_str=""):
    # Initialize all outputs in case everything fails
    return_code = None
    output      = None
    error_msg   = None

    shell_pre_flags = split_space(pre_flags_str)

    shell_bin_flag  = ["%s/%s" % (bin_path, bin_name)]

    shell_flags     = split_space(flags)

    shell_cmd = shell_pre_flags + shell_bin_flag + shell_flags

    try:
        # Start process with requested flags (and piped output from stdout & stderr)
        process = subprocess.Popen(shell_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Obtain output from communicate()
        output, unused_err = process.communicate(piped_input)

        # Poll for return code
        return_code = process.poll()

        # Error if return_code > 0
        if return_code:
            raise subprocess.CalledProcessError(return_code, flags, output=output)

    except Exception as e:
        # Grab output if there is any
        if('output' in e.__dict__):
            output = e.output

        # Grab return code if there is any
        if('returncode' in e.__dict__):
            return_code = e.returncode

        # Grab error (guaranteed to be given)
        error_msg = str(e)

    return RunResult(flags, bin_path, bin_name, output, error_msg, return_code, OutputParser(output)), shell_cmd



def create_full_cutlass_profiler_cmd_for_conv(kernel_specific_test_flag, parsed):
  kernel_specific_test_flag = Flags()
  # TODO: FILL THIS
  return kernel_specific_test_flag

def create_full_cutlass_profiler_cmd_for_gemm(kernel_specific_test_flag, parsed):
  kernel_specific_test_flag['cta-tile::m'] = (parsed['arguments'].cta_m,)
  kernel_specific_test_flag['cta-tile::n'] = (parsed['arguments'].cta_n,)
  kernel_specific_test_flag['cta-tile::k'] = (parsed['arguments'].cta_k,)
  kernel_specific_test_flag['warp-tile::m'] = (parsed['arguments'].warp_m,)
  kernel_specific_test_flag['warp-tile::n'] = (parsed['arguments'].warp_n,)
  kernel_specific_test_flag['warp-tile::k'] = (parsed['arguments'].warp_k,)
  kernel_specific_test_flag['split-count'] = (parsed['arguments'].split_k,)
  kernel_specific_test_flag['opcode-class'] = (parsed['arguments'].opcode_class,)
  kernel_specific_test_flag['accumulator-type'] = (parsed['arguments'].accumulator_type,)
  return kernel_specific_test_flag

def process_n_display_each_kernel_run(runlist_results, layer):
    flags = layer.flags
    for (output, parsed) in zip(runlist_results.outputs, runlist_results.parsed):
        # Print output if it exists
        if(output == None):
            print "No output detected\n"
        else:           
            kernel_specific_test_flag = Flags(True, None, None)
            if("Gemm" in parsed['schema'].name):
              kernel_specific_test_flag = create_full_cutlass_profiler_cmd_for_gemm(kernel_specific_test_flag, parsed)
            elif("Conv2d" in parsed['schema'].name):
              kernel_specific_test_flag = create_full_cutlass_profiler_cmd_for_conv(kernel_specific_test_flag, parsed)

            single_test_flag = flags.get_copy() + kernel_specific_test_flag
            # TODO: Use Arguments: in the parsed output to generate the cutlass profiler command to trigger this kernel
            print "&&&& RUNNING [ repro below kernel using: ./tools/profiler/cutlass_profiler %s ]" % single_test_flag.get_str()
            print "Layer Name: %s" % layer.base_name
            print "Test Flags: %s" % single_test_flag.get_str(prefix='', delimiter=' * ', seperator=':', quotify_comma=True)
            print "Unique Flags: %s" % kernel_specific_test_flag.get_str(prefix='', delimiter=' * ', seperator=':', quotify_comma=True)
            print output
            print "&&&& FINISHED ./tools/profiler/cutlass_profiler %s\n" % single_test_flag.get_str()

def overwrite_cutlass_profiler_flags_with_cmd_line(cutlass_flags, cutlass_cmd_line_flags_list):
  for cutlass_cmd_line_flags in cutlass_cmd_line_flags_list:
    cutlass_flags = cutlass_flags + cutlass_cmd_line_flags
  return cutlass_flags

class RunCache:
    def __init__(self, batch_size, max_cache):
        # Start with empty cache
        self.cache      = collections.OrderedDict()

        self.cache_count  = 0

        self.batch_size = int(batch_size)

        self.max_cache = int(max_cache)


    def run_cutlass_profiler(self, layer, bin_path, bin_name, piped_input=None, pre_flags_str=""):
        flags = layer.flags

        (flags, _, _, output, error_msg, return_code, _), shell_cmd = run_flags(flags, bin_path, bin_name, piped_input, pre_flags_str)

#        if return_code == 0:
#          (split_outputs, parsed_outputs) = parse_cutlass_profiler_output(output, bin_path, bin_name)
#          runlist_results = RunListResult(flags, bin_path, bin_name, split_outputs, error_msg, return_code, parsed_outputs)      
#          process_n_display_each_kernel_run(runlist_results, layer) #flags here is for a specific kernel. It make change if we launch many kernels for single use case
        return (output, error_msg, return_code), shell_cmd

    def get(self, flags, bin_path, bin_name, device='0', pre_flags_str=""):
        cache_key = flags.get_descs_str()

        while(cache_key not in self.cache):
            # Print out if a re-run was necessary
            self.run(flags, bin_path, bin_name, device=device, pre_flags_str=pre_flags_str)

        return self.cache[cache_key].run_result

    def exists(self, flags):
        if flags.get_descs_str() in self.cache:
            return True

        return False

    def insert(self, flags, run):
        cache_key = flags.get_descs_str()

        if cache_key in self.cache:
            self.cache[cache_key] = self.cache[cache_key]._replace(cache_count = self.cache_count)
            self.cache_count += 1
            return True

        while len(self.cache) == self.max_cache:
            # Pop first item from cache
            prev_first_key, prev_first_cached_run = self.cache.popitem(last=False)

            # Disallow removing any values that were run in this batch
            if self.cache_count - prev_first_cached_run.cache_count < self.batch_size:
                self.cache[prev_first_key] = prev_first_cached_run

        self.cache[flags.get_descs_str()] = CachedRunResult(self.cache_count, run)
        self.cache_count += 1
        return True

    # suggested_flags = [['-n10','-r3',...],['-n25','-r5',...],...]
    def update_suggested_flags(self, suggested_layers):
        self.suggested_flags = collections.OrderedDict()

        for layer in suggested_layers:
            flags = layer.flags

            key = flags.get_descs_str()

            if key in self.cache:
                self.cache[key] = self.cache[key]._replace(cache_count = self.cache_count)
                self.cache_count += 1

            elif key not in self.suggested_flags: # Not in cache (we have the result) not in the list
                self.suggested_flags[key] = flags.get_copy()
