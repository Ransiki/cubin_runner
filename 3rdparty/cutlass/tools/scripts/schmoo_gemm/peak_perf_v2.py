"""
This code is dervied from peak_perf.py, modified to schmoo across M, N and K for gv100 gemm kernels in CUTLASS
"""
import sys, csv, subprocess, argparse, os
import pandas as pd
from collections import OrderedDict

CSV_DIR = './assets/csv/'

#
def Statistics(series, percentile = 0.98):
  ''' Returns a tuple containing the best performance, the kth percentile, and the median. '''

  if len(series) == 0:
    return (0, 0, 0)

  sorted_series = sorted(series)

  max_result = max(sorted_series)
  best_result = sorted_series[int(len(sorted_series) * percentile)]
  median_result = sorted_series[int(0.5 * len(sorted_series))]

  return (max_result, best_result, median_result)


#
def PeakPerformance(file):
  ''' Given a csv document, extracts peak performance of CUTLASS results. '''

  series = []

  reader = csv.DictReader(file)
  for row in reader:
    if row['Provider'] == 'CUTLASS':
      gflops = float(row['GFLOPs'])
      series.append(gflops)

  return Statistics(series)

"""
The upper bound for m when schmooing m is a variable. The value depends on number of SMs
in a GPU, whose value is derived based on gpu-arch flag provided by the user
"""
schmoo_problem_size = {
"m" : "--problem-size::m=32:%s:32 --problem-size::n=256,1024,2048,4096 --problem-size::k=4096",
"n" : "--problem-size::n=32:%s:32 --problem-size::m=256,1024,2048,4096 --problem-size::k=4096",
"k" : "--problem-size::k=32:4096:32 --problem-size::m=256,1024,2048,%s --problem-size::n=4096"
}

quick_problem_size = "--problem-size::m=%s --problem-size::n=%s --problem-size::k=1024,2048,3072,4096,5120"

gemm_workloads = {
  'FFMA':
    [
      ('NN', '--A=f32:column --B=f32:column --accumulator-type=f32 --opcode-class=simt'),
      ('NT', '--A=f32:column --B=f32:row --accumulator-type=f32 --opcode-class=simt'),
      ('TN', '--A=f32:row --B=f32:column --accumulator-type=f32 --opcode-class=simt'),
      ('TT', '--A=f32:row --B=f32:row --accumulator-type=f32 --opcode-class=simt')
    ]
  ,
  'DFMA':
    [
      ('NN', '--A=f64:column --B=f64:column --accumulator-type=f64 --opcode-class=simt'),
      ('NT', '--A=f64:column --B=f64:row --accumulator-type=f64 --opcode-class=simt'),
      ('TN', '--A=f64:row --B=f64:column --accumulator-type=f64 --opcode-class=simt'),
      ('TT', '--A=f64:row --B=f64:row --accumulator-type=f64 --opcode-class=simt')
    ]
  ,
  'HFMA2':
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=simt'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=simt'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=simt'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=simt')
    ]
  ,
  'HMMA.884.F32':
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4')
    ]
  ,
  'HMMA.884.F16':
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4')
    ]
  ,
  'HMMA.1688.F32':
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8')
    ]
  ,
  'HMMA.1688.F16':
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8')
    ]
  ,
  'IMMA.8816':
    [
      ('NN', '--A=s8:column --B=s8:column --accumulator-type=s32 --opcode-class=tensorop --warp-tile::k=16'),
      ('NT', '--A=s8:column --B=s8:row --accumulator-type=s32 --opcode-class=tensorop --warp-tile::k=16'),
      ('TN', '--A=s8:row --B=s8:column --accumulator-type=s32 --opcode-class=tensorop --warp-tile::k=16'),
      ('TT', '--A=s8:row --B=s8:row --accumulator-type=s32 --opcode-class=tensorop --warp-tile::k=16')
    ]
}

SOL_ARCH = {
'tu102' :{
  'FFMA': 128,
  'DFMA': int(128 / 32),
  'HFMA2': 2 * 128,
  'HMMA.884.F32': 4 * 128,
  'HMMA.884.F16': 4 * 128,
  'HMMA.1688.F32': 8 * 128,
  'HMMA.1688.F16': 8 * 128,
  'IMMA.8816': 16 * 128,
},

'gv100' : {
  'FFMA': 128,
  'DFMA': int(128 / 2),
  'HFMA2': 2 * 128,
  'HMMA.884.F32': 4 * 128,
  'HMMA.884.F16': 4 * 128,
}
}

def create_sol_library(output_file_name, input_file_name):
  df = pd.read_csv(input_file_name)
  cutlass_df = df[df['Provider'] == 'CUTLASS']
  cutlass_df.reset_index(drop=True, inplace=True)
  cublas_df = df[df['Provider'] == 'cuBLAS']
  cublas_df.reset_index(drop=True, inplace=True)
  cutlass_df.loc[:,'cuBLAS GFLOPs'] = cublas_df['GFLOPs']
  cutlass_df.loc[:,'cuBLAS GB/s'] = cublas_df['GB/s']
  cutlass_df.loc[:,'cuBLAS MathUtilization'] = cublas_df['MathUtilization']
  cutlass_df.loc[:,'cuBLAS MemoryUtilization'] = cublas_df['MemoryUtilization']
  cutlass_df.to_csv(output_file_name)

def create_cutlass_csv(output_file_name, input_file_name):
  df = pd.read_csv(input_file_name)
  cutlass_df = df[df['Provider'] == 'CUTLASS']
  cutlass_df.to_csv(output_file_name)

# The output should be, the csv file should contain %SOL of both cublas and cutlass
def Profile(profiler_path, gpu_arch, cuda_device_index, result, name, layout, cmdline, schmoo, problem_size, num_sms, sm_clk, tags_dict, tags):
  output_name = CSV_DIR + "perf-%s-%s-%s-%s-all.csv" % (gpu_arch, name, layout, schmoo)
  output_cutlass_name = CSV_DIR + "perf-%s-%s-%s-%s.csv" % (gpu_arch, name, layout, schmoo)
  output_sol_name = CSV_DIR + "perf-%s-%s-%s-%s-sol.csv" % (gpu_arch, name, layout, schmoo)

  return_code = 0

  if not os.path.exists(output_name):
    cmdline_root = '%s/cutlass_profiler --function=Gemm --device=%s --tags=Op:%s,Layout:%s,%s --sleep-duration=5 --warmup-iterations=5 --clock=%s'  %  \
    (profiler_path, cuda_device_index, name, layout, tags, sm_clk)

    profiler_cmdline = "%s %s --output=%s %s" % (cmdline_root, problem_size, output_name, cmdline)

    return_code = subprocess.call(profiler_cmdline, shell=True)

  if os.path.exists(output_name) and not os.path.exists(output_cutlass_name):
    create_cutlass_csv(output_cutlass_name, output_name)

  if os.path.exists(output_name) and not os.path.exists(output_sol_name):
    create_sol_library(output_sol_name, output_name)

  with open(output_name, 'r') as file:
    statistics = PeakPerformance(file)

    max_performance, percentile, median_performance = statistics

    sol = SOL_ARCH[gpu_arch][name] * num_sms * sm_clk / 1000

    tags_dict['Instruction'] = name
    tags_dict['Layout'] = layout
    tags_dict['Max'] = str(max_performance)
    tags_dict['95th'] = str(percentile)
    tags_dict['Median'] = str(median_performance)
    tags_dict['SOL'] = str(sol)
    tags_dict['Utilization'] = str(percentile / sol)

    result.writerow(tags_dict)

  return return_code

# Add argparser for profiler path and gpu arch
def Main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu-arch', required=True, choices = ["gv100", "tu102"], help="gv100, tu102 are supported")
  parser.add_argument('--profiler-path', required=True, help="Provide path to cutlass_profiler")
  parser.add_argument('--cuda-device-index', required=True, help="Provide cuda device index for gpu to profile")
  parser.add_argument('--tags', help="Provide csv tag along with its value, eg: --tag \"nvcc version:10.0,mem clk:6500,sm clk:1500\"")
  parser.add_argument('--quick-mode', required=True, choices = [0, 1], type=bool, default=False, help="Run in quick mode or thorough mode")
  parser.add_argument('--num-sms', required=True, type=int, help="Number of SMs the gpu has")
  parser.add_argument('--sm-clock', required=True, type=int, help="SM clock in MHz")
  parser.add_argument('--memory-clock', required=True, type=int, help="Memory clock in MHz")

  args = parser.parse_args()

  profiler_path = args.profiler_path
  gpu_arch = args.gpu_arch
  cuda_device_index = args.cuda_device_index
  tags = args.tags
  num_sms = args.num_sms
  quick_mode = args.quick_mode
  sm_clock = args.sm_clock
  memory_clock = args.memory_clock

  tags = tags + ",SM Clock:%s,Memory Clock:%s" % (sm_clock, memory_clock)

  mkdir_cmds = ['mkdir', '-p', CSV_DIR]
  return_code = subprocess.run(mkdir_cmds)

  result_name = CSV_DIR + 'performance-result.csv'

  with open(result_name, 'w', newline='') as result_file:

    tags_dict = OrderedDict([ (tag.split(':')[0], tag.split(':')[1]) for tag in tags.split(',') ])

    fieldnames = list(tags_dict.keys()) + ['Instruction','Layout','Max','95th','Median','SOL','Utilization']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()

    if quick_mode == False:
      for name in SOL_ARCH[gpu_arch].keys():
        for layout, cmdline in gemm_workloads[name]:
          for schmoo_dim in ["m", "n", "k"]:
            problem_size = schmoo_problem_size[schmoo_dim] % (64 * num_sms)
            Profile(profiler_path, gpu_arch, cuda_device_index, writer, name, layout, cmdline, schmoo_dim, problem_size, num_sms, sm_clock, tags_dict, tags)
    else:
      for name in SOL_ARCH[gpu_arch].keys():
        for layout, cmdline in gemm_workloads[name]:
          problem_size = quick_problem_size % (64 * num_sms, 64 * num_sms)
          Profile(profiler_path, gpu_arch, cuda_device_index, writer, name, layout, cmdline, "k", problem_size, num_sms, sm_clock, tags_dict, tags)

  pass

#
if __name__ == "__main__":
  sys.exit(Main())

