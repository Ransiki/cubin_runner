#
#
#

import sys
import csv
import subprocess

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
  ''' Given a csv document, extracts peak performance of provider results. '''

  series = []

  reader = csv.DictReader(file)
  for row in reader:
    gflops = float(row['GFLOPs'])
    series.append(gflops)

  return Statistics(series)

#
#
#

#
gemm_workloads = [
  ('FFMA',
    1,
    [
      ('NN', '--A=f32:column --B=f32:column --accumulator-type=f32 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('NT', '--A=f32:column --B=f32:row --accumulator-type=f32 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('TN', '--A=f32:row --B=f32:column --accumulator-type=f32 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('TT', '--A=f32:row --B=f32:row --accumulator-type=f32 --opcode-class=simt --problem-size::k=2048,3072,4096')
    ]
  ),
  ('DFMA',
    1/2.0,
    [
      ('NN', '--A=f64:column --B=f64:column --accumulator-type=f64 --opcode-class=simt --problem-size::k=1024'),
      ('NT', '--A=f64:column --B=f64:row --accumulator-type=f64 --opcode-class=simt --problem-size::k=1024'),
      ('TN', '--A=f64:row --B=f64:column --accumulator-type=f64 --opcode-class=simt --problem-size::k=1024'),
      ('TT', '--A=f64:row --B=f64:row --accumulator-type=f64 --opcode-class=simt --problem-size::k=1024')
    ]
  ),
  ('HFMA2',
    2,
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=simt --problem-size::k=2048,3072,4096'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=simt --problem-size::k=2048,3072,4096')
    ]
  ),
  ('HMMA.884.F32',
    8,
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096')
    ]
  ),
  ('HMMA.884.F16',
    8,
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=4 --problem-size::k=2048,3072,4096')
    ]
  ),
  ('HMMA.1688.F32',
    8,
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f32 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512')
    ]
  ),
  ('HMMA.1688.F16',
    8,
    [
      ('NN', '--A=f16:column --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('NT', '--A=f16:column --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('TN', '--A=f16:row --B=f16:column --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512'),
      ('TT', '--A=f16:row --B=f16:row --accumulator-type=f16 --opcode-class=tensorop --warp-tile::k=8 --problem-size::k=2048:8192:512')
    ]
  ),
  ('IMMA.8816',
    8,
    [
      ('TN', '--A=s8:row --B=s8:column --accumulator-type=s32 --opcode-class=tensorop --warp-tile::k=16 --problem-size::k=2048:8192:512'),
    ]
  )
]

multiprocessors = 68
clock = 1.350

SOL = {
  'FFMA': multiprocessors * 128 * clock,
  'DFMA': multiprocessors * 128 / 32 * clock,
  'HFMA2': 2 * multiprocessors * 128 * clock,
  'HMMA.884.F32': multiprocessors * 4 * 128 * clock,
  'HMMA.884.F16': multiprocessors * 4 * 128 *  clock,
  'HMMA.1688.F32': multiprocessors * 8 * 128 * clock,
  'HMMA.1688.F16': multiprocessors * 8 * 128 * clock,
  'IMMA.8816': multiprocessors * 16 * 128 * clock,
}

#
def Profile(result, name, speedup, layout, cmdline, provider):

  profiler_path = './tools/profiler'
  gemm_m = 4352
  gemm_n = 4096
  k_start = 512
  k_end = 1024 + 256 * speedup

  output_name = "perf-%s-%s.csv" % (name, layout)

  cmdline_root = '%s/cutlass_profiler --function=Gemm --clock=1350 --tags=jetfire:disabled,chip:tu102,clock:1350,branch:2.x --providers=%s --sleep-duration=5 --warmup-iterations=5 --problem-size::m=%d --problem-size::n=%d ' %  \
    (profiler_path, provider, gemm_m, gemm_n)

  profiler_cmdline = "%s --output=%s %s" % (cmdline_root, output_name, cmdline)

  return_code = subprocess.call(profiler_cmdline, shell=True)

  with open(output_name, 'r') as file:
    statistics = PeakPerformance(file)

    max_performance, percentile, median_performance = statistics

    sol = SOL[name]

    result.writerow({
      'Provider': provider,
      'Instruction': name,
      'Layout': layout,
      'Max': str(max_performance),
      '95th': str(percentile),
      'Median': str(median_performance),
      'SOL': str(sol),
      'Utilization': str(percentile / sol)
    })

  return return_code

#
def Main():

  result_name = 'performance-result.csv'

  with open(result_name, 'w', newline='') as result_file:

    fieldnames = ['Provider', 'Instruction','Layout','Max','95th','Median','SOL','Utilization']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()

    for workload in gemm_workloads:
      name, speedup, layouts = workload
      for layout, cmdline in layouts:
        Profile(writer, name, speedup, layout, cmdline, "cutlass")
        Profile(writer, name, speedup, layout, cmdline, "cublas")

  pass

#
if __name__ == "__main__":
  sys.exit(Main())
