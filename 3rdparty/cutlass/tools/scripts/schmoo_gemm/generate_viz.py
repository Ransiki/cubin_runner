import pandas as pd
import subprocess as sp
import argparse

OPS = {
'gv100' : ['FFMA', 'HFMA2', 'DFMA', 'HMMA.884.F16', 'HMMA.884.F32'],
'tu102' : ['FFMA', 'HFMA2', 'DFMA', 'HMMA.884.F16', 'HMMA.884.F32', 'HMMA.1688.F16', 'HMMA.1688.F32', 'IMMA.8816']
}

MAX_DIM = {
'tu102' : 64 * 68,
'gv100' : 64 * 80
}

LAYOUTS = ['NN', 'NT', 'TN', 'TT']
CSV_DIR = './assets/csv/'
def main(gpu_arch):
  output_dir = './assets/data/'
  cmds = ['mkdir', '-p', output_dir]
  sp.run(cmds)
  print("Successfully created directory " + output_dir)
  files = []
  for ops in OPS[gpu_arch]:
    layouts = LAYOUTS
    if ops == 'IMMA.8816':
      layouts = ['TN']
    js_file_name = "schmoo_" + gpu_arch + "_" + ops.lower().replace('.','_') + ".js"
    csv_data, csv_sol_data = "", ""
    for layout in layouts:
      csv_file_name = CSV_DIR + "perf-%s-%s-%s-%s-sol.csv" %(gpu_arch, ops, layout, "m")
      print(csv_file_name)
      df = pd.read_csv(csv_file_name)
      js_function_name = "data_" + '_'.join([ops.lower().replace('.','_'), layout, "m"])
      js_sol_function_name = js_function_name + "_sol"
      m_range = df['problem-size::m'].unique()
      n_range = [256, 1024, 2048, 4096]
      k_range = [4096]
      csv_data = csv_data + "function " + js_function_name + "(){ return \"M,N - 256,N - 1024,N - 2048"
      csv_sol_data = csv_sol_data + "function " + js_sol_function_name + "(){ return \"M,CUTLASS SOL,cuBLAS SOL"
      for m in m_range:
        csv_data = csv_data + "\\n\\\n" + str(m)
        csv_sol_data = csv_sol_data + "\\n\\\n"
        for n in n_range:
          for k in k_range:
            df_m = df[df['problem-size::m'] == m]
            df_n = df_m[df_m['problem-size::n'] == n]
            max_perf = df_n[df_n['problem-size::k'] == k]['GFLOPs'].max()
            max_perf_index = df_n[df_n['GFLOPs'] == max_perf].index[0]
            cublas_sol = df.iloc[max_perf_index]['cuBLAS MathUtilization']
            cutlass_sol = df.iloc[max_perf_index]['MathUtilization']
            csv_data = csv_data + "," + str(cutlass_sol)
            if n == 4096:
              csv_sol_data = csv_sol_data + ",".join([str(m), str(cutlass_sol), str(cublas_sol)])
      csv_data = csv_data + "\"}\n\n"
      csv_sol_data = csv_sol_data + "\"}\n\n"

      csv_file_name = CSV_DIR + "perf-%s-%s-%s-%s-sol.csv" %(gpu_arch, ops, layout, "n")
      print(csv_file_name)
      df = pd.read_csv(csv_file_name)
      js_function_name = "data_" + '_'.join([ops.lower().replace('.','_'), layout, "n"])
      js_sol_function_name = js_function_name + "_sol"
      n_range = df['problem-size::n'].unique()
      m_range = [256, 1024, 2048, 4096]
      k_range = [4096]
      csv_data = csv_data + "function " + js_function_name + "(){ return \"N,M - 256,M - 1024,M - 2048"
      csv_sol_data = csv_sol_data + "function " + js_sol_function_name + "(){ return \"N,CUTLASS SOL,cuBLAS SOL"
      for n in n_range:
        csv_data = csv_data + "\\n\\\n" + str(n)
        csv_sol_data = csv_sol_data + "\\n\\\n"
        for m in m_range:
          for k in k_range:
            df_n = df[df['problem-size::n'] == n]
            df_m = df_n[df_n['problem-size::m'] == m]
            max_perf = df_m[df_m['problem-size::k'] == k]['GFLOPs'].max()
            max_perf_index = df_m[df_m['GFLOPs'] == max_perf].index[0]
            cublas_sol = df.iloc[max_perf_index]['cuBLAS MathUtilization']
            cutlass_sol = df.iloc[max_perf_index]['MathUtilization']
            csv_data = csv_data + "," + str(cutlass_sol)
            if m == 4096:
              csv_sol_data = csv_sol_data + ",".join([str(n), str(cutlass_sol), str(cublas_sol)])
      csv_data = csv_data + "\"}\n\n"
      csv_sol_data = csv_sol_data + "\"}\n\n"

      csv_file_name = CSV_DIR + "perf-%s-%s-%s-%s-sol.csv" %(gpu_arch, ops, layout, "k")
      print(csv_file_name)
      df = pd.read_csv(csv_file_name)
      js_function_name = "data_" + '_'.join([ops.lower().replace('.','_'), layout, "k"])
      js_sol_function_name = js_function_name + "_sol"
      k_range = df['problem-size::k'].unique()
      m_range = [256, 1024, 2048] + [MAX_DIM[gpu_arch]]
      n_range = [4096]
      csv_data = csv_data + "function " + js_function_name + "(){ return \"K,M - 256,M - 1024,M - 2048,M - 4096"
      csv_sol_data = csv_sol_data + "function " + js_sol_function_name + "(){ return \"K,CUTLASS SOL,cuBLAS SOL"
      for k in k_range:
        csv_data = csv_data + "\\n\\\n" + str(k)
        csv_sol_data = csv_sol_data + "\\n\\\n"
        for m in m_range:
          for n in n_range:
            df_k = df[df['problem-size::k'] == k]
            df_m = df_k[df_k['problem-size::m'] == m]
            max_perf = df_m[df_m['problem-size::n'] == n]['GFLOPs'].max()
            max_perf_index = df_m[df_m['GFLOPs'] == max_perf].index[0]
            cublas_sol = df.iloc[max_perf_index]['cuBLAS MathUtilization']
            cutlass_sol = df.iloc[max_perf_index]['MathUtilization']
            csv_data = csv_data + "," + str(cutlass_sol)
            if m == MAX_DIM[gpu_arch]:
              csv_sol_data = csv_sol_data + ",".join([str(k), str(cutlass_sol), str(cublas_sol)])
      csv_data = csv_data + "\"}\n\n"
      csv_sol_data = csv_sol_data + "\"}\n\n"

    with open(output_dir + js_file_name, "w+") as f:
      f.write(csv_data + csv_sol_data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu-arch", required=True, help="gv100 or tu102")
  args = parser.parse_args()
  gpu_arch = args.gpu_arch
  if gpu_arch != 'gv100' and gpu_arch != 'tu102':
    print("Enter valid gpu arch - gv100 or tu102. Got: ", gpu_arch)
    exit()
  main(gpu_arch)
