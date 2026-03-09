"""
This file contains code to capture output from profiler and load into a panda data frame.
We load it until certain threshold and write it to a single file
"""

CSV_RESULTS_NAME = 'CSV Results:' # Used for getting csv
CUTLASS_CSV_HEADER_GEMM = 'Problem,Provider,Schema,Type,Function,Disposition,Status,problem-size::m,problem-size::n,problem-size::k,A,B,C,D,epilogue::alpha,epilogue::beta,split-count,opcode-class,accumulator-type,cta-tile::m,cta-tile::n,cta-tile::k,warp-tile::m,warp-tile::n,warp-tile::k,cuBLAS,cuDNN,Bytes,Flops,Runtime,GB/s,GFLOPs'
CUTLASS_CSV_HEADER_CONV = 'Problem,Provider,Schema,Type,Function,Disposition,Status,input-size::n,input-size::h,input-size::w,input-size::c,filter-size::k,filter-size::r,filter-size::s,filter-size::c,Activation,Filter,Output,epilogue::alpha,epilogue::beta,split-count,pad::top,pad::bottom,pad::left,pad::right,stride::h,stride::w,dilation::h,dilation::w,opcode-class,accumulator-type,conv-mode,cta-tile::m,cta-tile::n,cta-tile::k,warp-tile::m,warp-tile::n,warp-tile::k,op-tile::m,op-tile::n,op-tile::k,cuBLAS,cuDNN,Bytes,Flops,Runtime,GB/s,GFLOPs'
GPU_POWER_PROFILE_FILENAME = './helpers/power_gpu.csv'

import pandas as pd

def get_gpu_power_profiles():
  with open(GPU_POWER_PROFILE_FILENAME) as f:
    df = pd.read_csv(f)
    return df

class log_csv():
  def __init__(self, file_name, layer_type):
    self.all_df_filename = file_name + "_" + layer_type + "_all.csv"
    self.bad_return_df_filename = file_name + "_" + layer_type + "_bad_return.csv"
    self.no_output_df_filename = file_name + "_" + layer_type + "_no_output.csv"
    self.all_df_start, self.bad_return_df_start, self.no_output_df_start = False, False, False
    if layer_type == 'gemm':
      self.all_df_header = ["Layer Name", "Cmd", "Output"] + CUTLASS_CSV_HEADER_GEMM.split(',')
      self.all_commas = [''] * len(CUTLASS_CSV_HEADER_GEMM.split(','))
    else:
      self.all_df_header = ["Layer Name", "Cmd", "Output"] + CUTLASS_CSV_HEADER_CONV.split(',')
      self.all_commas = [''] * len(CUTLASS_CSV_HEADER_CONV.split(','))
    self.bad_return_df_header = ["Layer Name", "Cmd"]
    self.no_output_df_header = ["Layer Name", "Cmd"]
    self.iteration = 0
  def add_to_log(self, layer_name, shell_cmd, raw_data, return_code):
    cmd = ' '.join(shell_cmd)
    if return_code == 0 and raw_data != "":
      raw_csv = raw_data.split(CSV_RESULTS_NAME)[-1]
      csv = raw_csv.split('\n')
      data = csv[3:-1]
      rows = [[layer_name, cmd, 'good'] + row.split(',') for row in data]
      if self.all_df_start == False:
        self.all_df = pd.DataFrame(rows, columns = self.all_df_header)
        self.all_df_start = True
      else:
        self.all_df = pd.concat([self.all_df, pd.DataFrame(rows, columns = self.all_df_header)])
    elif return_code != 0:
      if self.bad_return_df_start == False:
        self.bad_return_df = pd.DataFrame([[layer_name, cmd]], columns = self.bad_return_df_header)
        self.bad_return_df_start = True
      else:
        self.bad_return_df = pd.concat([self.bad_return_df, pd.DataFrame([(layer_name, cmd)], columns = self.bad_return_df_header)])
      if self.all_df_start == False:
        self.all_df = pd.DataFrame([layer_name,cmd,'bad_return'] + self.all_commas, columns = self.all_df_header)
        self.all_df_start = True
      else:
        self.all_df = pd.concat([self.all_df, pd.DataFrame([[layer_name,cmd,'bad_return'] + self.all_commas], columns = self.all_df_header)])
    else:
      if self.no_output_df_start == False:
        self.no_output_df = pd.DataFrame([(layer_name, cmd)], columns = self.no_output_df_header)
        self.no_output_df_start = True
      else:
        self.no_output_df = pd.concat([self.no_output_df, pd.DataFrame([(layer_name, cmd)], columns = self.no_output_df_header)])
      if self.all_df_start == False:
        data = [layer_name, cmd, 'no_output'] + self.all_commas
        print len(data), len(self.all_df_header)
        self.all_df = pd.DataFrame([data], columns = self.all_df_header)
        self.all_df_start = True
      else:
        data = [layer_name,cmd, 'no_output'] + self.all_commas
        self.all_df = pd.concat([self.all_df, pd.DataFrame([data], columns = self.all_df_header)])
      # For every 10 iterations, write to file to save intermediate data
      if self.iteration % 10 == 0:
        if self.all_df_start == True:
          self.all_df.to_csv(self.all_df_filename, index = False)
        if self.bad_return_df_start == True:
          self.bad_return_df.to_csv(self.bad_return_df_filename, index = False)
        if self.no_output_df_start == True:
          self.no_output_df.to_csv(self.no_output_df_filename, index = False)
  def print_log(self):
    if self.all_df_start == True:
      print self.all_df
    if self.bad_return_df_start == True:
      print self.bad_return_df
    if self.no_output_df_start == True:
      print self.no_output_df
  def write_to_csv(self):
    if self.all_df_start == True:
      self.all_df.to_csv(self.all_df_filename, index = False)
    if self.bad_return_df_start == True:
      self.bad_return_df.to_csv(self.bad_return_df_filename, index = False)
    if self.no_output_df_start == True:
      self.no_output_df.to_csv(self.no_output_df_filename, index = False)
