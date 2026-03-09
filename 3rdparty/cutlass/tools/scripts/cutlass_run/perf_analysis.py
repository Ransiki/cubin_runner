"""CUTLASS profiling data visualization tool

This module takes csv file from user and generate graphs depending on the type of file
"""

import argparse
import pandas as pd

f_heading_html = lambda label : "<th>" + str(label) + "</th>\n"
f_data_html = lambda data : "<td>" + str(data) + "</td>\n"

ASSETS_DIR = "./assets/data/"

def generate_conv_data(conv_df, conv_type):
  conv_cmds_dict = {}
  conv_cmds = list(conv_df['Cmd'].unique())
  conv_cutlass_max_number_of_outputs = 0

  for cmd in conv_cmds:
    cmds_df = conv_df[conv_df['Cmd'] == cmd]
    cmd_dict = {'cutlass_row' : [], 'cudnn_row' : [], 'cutlass_index' : [], 'cudnn_index' : [], 'cutlass_perf' : [], 'cudnn_perf' : [], 'best_cutlass_perf' : 0, 'best_cudnn_perf' : 0, 'best_cutlass_perf_index' : -1, 'best_cudnn_perf_index' : -1, 'layer_name' : '', 'relative_perf' : 0, 'best_cutlass_perf_row' : ''}
    for index, row in cmds_df.iterrows():
      row_str = ';'.join(str(x) for x in row.values)
      if row['Provider'] == 'CUTLASS':
        cmd_dict['cutlass_index'].append(index)
        cmd_dict['cutlass_perf'].append(row['GFLOPs'])
        cmd_dict['cutlass_row'].append(row_str)
        cmd_dict['best_cutlass_perf'] = max(cmd_dict['best_cutlass_perf'], row['GFLOPs'])
      if row['Provider'] == 'cuDNN':
        cmd_dict['cudnn_index'].append(index)
        cmd_dict['cudnn_perf'].append(row['GFLOPs'])
        cmd_dict['cudnn_row'].append(row_str)
        cmd_dict['best_cudnn_perf'] = max(cmd_dict['best_cudnn_perf'], row['GFLOPs'])
      cmd_dict['layer_name'] = row['Layer Name']
    if len(cmd_dict['cutlass_perf']) > 0:
      cmd_dict['best_cutlass_perf_index'] = cmd_dict['cutlass_index'][cmd_dict['cutlass_perf'].index(cmd_dict['best_cutlass_perf'])]
      cmd_dict['best_cutlass_perf_row'] = cmd_dict['cutlass_row'][cmd_dict['cutlass_perf'].index(cmd_dict['best_cutlass_perf'])]
      conv_cmds_dict[cmd] = cmd_dict
    if len(cmd_dict['cudnn_perf']) > 0:
      cmd_dict['best_cudnn_perf_index'] = cmd_dict['cudnn_index'][cmd_dict['cudnn_perf'].index(cmd_dict['best_cudnn_perf'])]
      conv_cmds_dict[cmd] = cmd_dict
    conv_cutlass_max_number_of_outputs = max(conv_cutlass_max_number_of_outputs, len(cmd_dict['cutlass_perf']))

  conv_rel_perf_columns = ["Row Index", "Layer Name", "Relative Performance"]
  conv_rel_perf_df = pd.DataFrame(columns=conv_rel_perf_columns)
  j = 0
  for cmd in conv_cmds:
    if cmd in conv_cmds_dict.keys():
      len_cutlass_perf = len(conv_cmds_dict[cmd]['cutlass_perf'])
      if len_cutlass_perf > 0:
        for i in range(conv_cutlass_max_number_of_outputs - len_cutlass_perf):
          conv_cmds_dict[cmd]['cutlass_perf'].append(0)
          conv_cmds_dict[cmd]['cutlass_row'].append("")
          conv_cmds_dict[cmd]['cutlass_index'].append(-1)
      len_cudnn_perf = len(conv_cmds_dict[cmd]['cudnn_perf'])
      if len_cudnn_perf > 0:
        for i in range(conv_cutlass_max_number_of_outputs - len_cudnn_perf):
          conv_cmds_dict[cmd]['cudnn_perf'].append(0)
          conv_cmds_dict[cmd]['cudnn_row'].append("")
          conv_cmds_dict[cmd]['cudnn_index'].append(-1)
      conv_cmds_dict[cmd]['relative_perf'] = conv_cmds_dict[cmd]['best_cutlass_perf'] / conv_cmds_dict[cmd]['best_cudnn_perf']
      conv_rel_perf_df = pd.concat([pd.DataFrame([[conv_cmds_dict[cmd]['best_cutlass_perf_index'], conv_cmds_dict[cmd]['layer_name'], conv_cmds_dict[cmd]['relative_perf']]], columns=conv_rel_perf_columns), conv_rel_perf_df]) 
    j = j + 1
  conv_rel_perf_df = conv_rel_perf_df.sort_values(by=['Relative Performance'])
  conv_rel_perf_df.reset_index(drop=True, inplace=True)
    
  conv_cutlass_perf_labels = ['CUTLASS Perf ' + str(i + 1) for i in range(conv_cutlass_max_number_of_outputs)]
  conv_cudnn_perf_labels = ['cuDNN Perf' + str(i + 1) for i in range(conv_cutlass_max_number_of_outputs)]
    
  conv_csv_string = "Index,Layer Name," + ','.join(conv_cutlass_perf_labels)
  conv_csv_max_perf_string = "Index,Best CUTLASS Performance,Best cuBLAS Performance"
  conv_csv_rel_perf_string = "Index,Relative Performance"

  js_table_conv_perf_string = "<tr>\n" + f_heading_html("Problem Id") + f_heading_html("Index") + ''.join([f_heading_html(x) for x in conv_df.columns.to_list()]) + "</tr>\n"

  js_table_conv_max_perf_string = "<tr>\n<th>Problem Id</th>\n\
        <th>Layer Name</th>\n\
        <th>Command</th>\n\
        <th>Best CUTLASS Perf Row Index</th>\n\
        <th>Best cuDNNN Perf Row Index</th>\n\
        <th>Best CUTLASS Performance</th>\n\
        <th>Best cuDNN Performance</th>\n\
        <th>Row</th>\n\
        </tr>\n"

  js_table_conv_rel_perf_string = "<tr>\n<th>Problem Id</th>\n"
  for value in conv_rel_perf_df.columns.to_list():
    js_table_conv_rel_perf_string = js_table_conv_rel_perf_string + "<th>\n" + str(value) + "</th>\n"
  js_table_conv_rel_perf_string = js_table_conv_rel_perf_string + "</tr>"

  for index, row in conv_df.iterrows():
    problem_id = conv_cmds.index(str(row['Cmd']))
    js_table_conv_perf_string = js_table_conv_perf_string + "<tr>\n" + f_data_html(str(problem_id)) +  f_data_html(str(index)) + ''.join([f_data_html(x) for x in row.to_list()]) + "</tr>\n"

  j = 0
  for cmd in conv_cmds:
    if cmd in conv_cmds_dict.keys():
      conv_csv_string = conv_csv_string + '\\n\\n' + ','.join([str(j)] + [str(x) for x in conv_cmds_dict[cmd]['cutlass_perf']])
    else:
      conv_csv_string = conv_csv_string + '\\n\\n' + ','.join([str(j)] + ['0'] * conv_cutlass_max_number_of_outputs)
    j = j + 1

  j = 0
  for cmd in conv_cmds:
    if cmd in conv_cmds_dict.keys():
      conv_csv_max_perf_string = conv_csv_max_perf_string + '\\n\\n' + ','.join([str(j), str(conv_cmds_dict[cmd]['best_cutlass_perf']), str(conv_cmds_dict[cmd]['best_cudnn_perf'])])
      js_table_conv_max_perf_string = js_table_conv_max_perf_string + \
            "<td>" + str(j) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['layer_name']) + "</td>\n" + \
            "<td>" + str(cmd) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['best_cutlass_perf_index']) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['best_cudnn_perf_index']) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['best_cutlass_perf']) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['best_cudnn_perf']) + "</td>\n" + \
            "<td>" + str(conv_cmds_dict[cmd]['best_cutlass_perf_row']) + "</td>\n" + \
            "</tr>\n"
    j = j + 1

  for index, row in conv_rel_perf_df.iterrows():
    js_table_conv_rel_perf_string = js_table_conv_rel_perf_string + "<tr>\n<td>" + str(index) + "</td>\n"
    for value in row.to_list():
      js_table_conv_rel_perf_string = js_table_conv_rel_perf_string + "<td>" + str(value) + "</td>\n"
    js_table_conv_rel_perf_string = js_table_conv_rel_perf_string + "</tr>\n"
    conv_csv_rel_perf_string = conv_csv_rel_perf_string + '\\n\\\n' + ','.join([str(index), str(row['Relative Performance'])])

  js_table_conv_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_conv_perf_string + "</table>\n</body>\n</html>"
  js_table_conv_rel_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_conv_rel_perf_string + "</table>\n</body>\n</html>"
  js_table_conv_max_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_conv_max_perf_string + "</table>\n</body>\n</html>"

  conv_csv_rel_perf_string = "function data_rel_perf_" + conv_type + "() {\n return \"" + conv_csv_rel_perf_string + "\"\n}"
  conv_csv_string = "function data_" + conv_type + "() {\n return \"" + conv_csv_string + "\"\n}"
  conv_csv_max_perf_string = "function data_max_perf_" + conv_type + "(){\n return \"" + conv_csv_max_perf_string + "\"\n}"

  with open(ASSETS_DIR + conv_type + "_data_max_perf.js", "w+") as f:
    f.write(conv_csv_max_perf_string)
  with open(ASSETS_DIR + conv_type + "_data.js", "w+") as f:
    f.write(conv_csv_string)
  with open(ASSETS_DIR + conv_type + "_data_rel_perf.js", "w+") as f:
    f.write(conv_csv_rel_perf_string)

  with open(ASSETS_DIR + conv_type + "_table_data_perf.html", "w+") as f:
    f.write(js_table_conv_perf_string)
  with open(ASSETS_DIR + conv_type + "_table_data_rel_perf.html", "w+") as f:
    f.write(js_table_conv_rel_perf_string)
  with open(ASSETS_DIR + conv_type + "_table_data_max_perf.html", "w+") as f:
    f.write(js_table_conv_max_perf_string)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv_file',
    type=str,
    dest='csv_file',
    help="The files passed should following below naming format \
      <layer file name>_<time stamp>_<status>.csv \
      <status> = all/no output/bad return"
  )

  args = parser.parse_args()

  ref_ops = {'conv' : 'cuDNN', 'gemm' : 'cuBLAS'}

  print args.csv_file
  all_file = False if args.csv_file.find('all') == -1 else True
  bad_return_file = False if args.csv_file.find('bad_return') == -1 else True
  no_output_file = False if args.csv_file.find('no_output') == -1 else True
  print all_file, bad_return_file, no_output_file
  file_name = args.csv_file
  if file_name.find('conv') == -1 and file_name.find('gemm') == -1:
    print "Pass a file with conv or gemm in their file name"
    exit()
  op = 'conv' if file_name.find('conv') != -1 else 'gemm'
  if all_file:
    if op == 'conv':
      df = pd.read_csv(file_name)
      fprop_df = pd.DataFrame(columns=df.columns)
      dgrad_df = pd.DataFrame(columns=df.columns)
      wgrad_df = pd.DataFrame(columns=df.columns)
      for index, row in df.iterrows():
        if row['Cmd'].find('Fprop') != -1:
          fprop_df = pd.concat([fprop_df, pd.DataFrame([row], columns=df.columns)])
        if row['Cmd'].find('Dgrad') != -1:
          dgrad_df = pd.concat([dgrad_df, pd.DataFrame([row], columns=df.columns)])
        if row['Cmd'].find('Wgrad') != -1:
          wgrad_df = pd.concat([wgrad_df, pd.DataFrame([row], columns=df.columns)])
      fprop_df.reset_index(drop=True, inplace=True)
      dgrad_df.reset_index(drop=True, inplace=True)
      wgrad_df.reset_index(drop=True, inplace=True)
      generate_conv_data(fprop_df, "fprop")
      generate_conv_data(dgrad_df, "dgrad")
      generate_conv_data(wgrad_df, "wgrad")
    else:
      gemm_df = pd.read_csv(file_name)
      cmds = list(gemm_df['Cmd'].unique())
      cmds_dict = {}
      cutlass_max_number_of_outputs = 0

      for cmd in cmds:
        cmds_df = gemm_df[gemm_df['Cmd'] == cmd]
        cmd_dict = {'cutlass_row' : [], 'cublas_row' : [], 'cutlass_index' : [], 'cublas_index' : [], 'cutlass_perf' : [], 'cublas_perf' : [], 'best_cutlass_perf' : 0, 'best_cublas_perf' : 0, 'best_cutlass_perf_index' : -1, 'best_cublas_perf_index' : -1, 'layer_name' : '', 'relative_perf' : 0, 'best_cutlass_perf_row' : ''}
        for index, row in cmds_df.iterrows():
          row_str = ';'.join(str(x) for x in row.values)
          if row['Provider'] == 'CUTLASS':
            cmd_dict['cutlass_index'].append(index)
            cmd_dict['cutlass_perf'].append(row['GFLOPs'])
            cmd_dict['cutlass_row'].append(row_str)
            cmd_dict['best_cutlass_perf'] = max(cmd_dict['best_cutlass_perf'], row['GFLOPs'])
          if row['Provider'] == 'cuBLAS':
            cmd_dict['cublas_index'].append(index)
            cmd_dict['cublas_perf'].append(row['GFLOPs'])
            cmd_dict['cublas_row'].append(row_str)
            cmd_dict['best_cublas_perf'] = max(cmd_dict['best_cublas_perf'], row['GFLOPs'])
          cmd_dict['layer_name'] = row['Layer Name']
        if len(cmd_dict['cutlass_perf']) > 0:
          cmd_dict['best_cutlass_perf_index'] = cmd_dict['cutlass_index'][cmd_dict['cutlass_perf'].index(cmd_dict['best_cutlass_perf'])]
          cmd_dict['best_cutlass_perf_row'] = cmd_dict['cutlass_row'][cmd_dict['cutlass_perf'].index(cmd_dict['best_cutlass_perf'])]
          cmds_dict[cmd] = cmd_dict
        if len(cmd_dict['cublas_perf']) > 0:
          cmd_dict['best_cublas_perf_index'] = cmd_dict['cublas_index'][cmd_dict['cublas_perf'].index(cmd_dict['best_cublas_perf'])]
          cmds_dict[cmd] = cmd_dict
        cutlass_max_number_of_outputs = max(cutlass_max_number_of_outputs, len(cmd_dict['cutlass_perf']))

      gemm_rel_perf_columns = ["Row Index", "Layer Name", "Relative Performance"]
      gemm_rel_perf_df = pd.DataFrame(columns=gemm_rel_perf_columns)
      j = 0
      for cmd in cmds:
        if cmd in cmds_dict.keys():
          len_cutlass_perf = len(cmds_dict[cmd]['cutlass_perf'])
          if len_cutlass_perf > 0:
            for i in range(cutlass_max_number_of_outputs - len_cutlass_perf):
              cmds_dict[cmd]['cutlass_perf'].append(0)
              cmds_dict[cmd]['cutlass_row'].append("")
              cmds_dict[cmd]['cutlass_index'].append(-1)
          len_cublas_perf = len(cmds_dict[cmd]['cublas_perf'])
          if len_cublas_perf > 0:
            for i in range(cutlass_max_number_of_outputs - len_cublas_perf):
              cmds_dict[cmd]['cublas_perf'].append(0)
              cmds_dict[cmd]['cublas_row'].append("")
              cmds_dict[cmd]['cublas_index'].append(-1)
          cmds_dict[cmd]['relative_perf'] = cmds_dict[cmd]['best_cutlass_perf'] / cmds_dict[cmd]['best_cublas_perf']
          gemm_rel_perf_df = pd.concat([pd.DataFrame([[cmds_dict[cmd]['best_cutlass_perf_index'], cmds_dict[cmd]['layer_name'], cmds_dict[cmd]['relative_perf']]], columns=gemm_rel_perf_columns), gemm_rel_perf_df])
        j = j + 1

      gemm_rel_perf_df = gemm_rel_perf_df.sort_values(by=['Relative Performance'])
      gemm_rel_perf_df.reset_index(drop=True, inplace=True)

      cutlass_perf_labels = ['CUTLASS Perf ' + str(i + 1) for i in range(cutlass_max_number_of_outputs)]
      cublas_perf_labels = ['cuBLAS Perf' + str(i + 1) for i in range(cutlass_max_number_of_outputs)]

      gemm_csv_string = "Index,Layer Name," + ','.join(cutlass_perf_labels)
      gemm_csv_max_perf_string = "Index,Best CUTLASS Performance,Best cuBLAS Performance"
      gemm_csv_rel_perf_string = "Index,Relative Performance"

      heading_html = lambda label : "<th>" + str(label) + "</th>\n"
      data_html = lambda data : "<td>" + str(data) + "</td>\n"

      js_table_gemm_perf_string = "<tr>\n" + heading_html("Problem Id") + heading_html("Index") + ''.join([heading_html(x) for x in gemm_df.columns.to_list()]) + "</tr>\n"

      js_table_gemm_max_perf_string = "<tr>\n<th>Problem Id</th>\n\
        <th>Layer Name</th>\n\
        <th>Command</th>\n\
        <th>Best CUTLASS Perf Row Index</th>\n\
        <th>Best cuBLAS Perf Row Index</th>\n\
        <th>Best CUTLASS Performance</th>\n\
        <th>Best cuBLAS Performance</th>\n\
        <th>Row</th>\n\
        </tr>\n"

      js_table_gemm_rel_perf_string = "<tr>\n<th>Problem Id</th>\n"
      for value in gemm_rel_perf_df.columns.to_list():
        js_table_gemm_rel_perf_string = js_table_gemm_rel_perf_string + "<th>\n" + str(value) + "</th>\n"
      js_table_gemm_rel_perf_string = js_table_gemm_rel_perf_string + "</tr>"

      for index, row in gemm_df.iterrows():
        problem_id = cmds.index(str(row['Cmd']))
        js_table_gemm_perf_string = js_table_gemm_perf_string + "<tr>\n" + data_html(str(problem_id)) +  data_html(str(index)) + ''.join([data_html(x) for x in row.to_list()]) + "</tr>\n"

      j = 0
      for cmd in cmds:
        if cmd in cmds_dict.keys():
          gemm_csv_string = gemm_csv_string + '\\n\\n' + ','.join([str(j)] + [str(x) for x in cmds_dict[cmd]['cutlass_perf']])
        else:
          gemm_csv_string = gemm_csv_string + '\\n\\n' + ','.join(['0'] * (cutlass_max_number_of_outputs + 1))
        j = j + 1

      j = 0
      for cmd in cmds:
        if cmd in cmds_dict.keys():
          gemm_csv_max_perf_string = gemm_csv_max_perf_string + '\\n\\n' + ','.join([str(j), str(cmds_dict[cmd]['best_cutlass_perf']), str(cmds_dict[cmd]['best_cublas_perf'])])
          js_table_gemm_max_perf_string = js_table_gemm_max_perf_string + \
            "<td>" + str(j) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['layer_name']) + "</td>\n" + \
            "<td>" + str(cmd) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['best_cutlass_perf_index']) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['best_cublas_perf_index']) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['best_cutlass_perf']) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['best_cublas_perf']) + "</td>\n" + \
            "<td>" + str(cmds_dict[cmd]['best_cutlass_perf_row']) + "</td>\n" + \
            "</tr>\n"
          j = j + 1

      for index, row in gemm_rel_perf_df.iterrows():
        js_table_gemm_rel_perf_string = js_table_gemm_rel_perf_string + "<tr>\n<td>" + str(index) + "</td>\n"
        for value in row.to_list():
          js_table_gemm_rel_perf_string = js_table_gemm_rel_perf_string + "<td>" + str(value) + "</td>\n"
        js_table_gemm_rel_perf_string = js_table_gemm_rel_perf_string + "</tr>\n"
        gemm_csv_rel_perf_string = gemm_csv_rel_perf_string + '\\n\\\n' + ','.join([str(index), str(row['Relative Performance'])])

      js_table_gemm_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_gemm_perf_string + "</table>\n</body>\n</html>"
      js_table_gemm_rel_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_gemm_rel_perf_string + "</table>\n</body>\n</html>"
      js_table_gemm_max_perf_string = "<html>\n<body>\n<table style=\"width:100%\">" + js_table_gemm_max_perf_string + "</table>\n</body>\n</html>"

      gemm_csv_rel_perf_string = "function data_rel_perf_gemm() {\n return \"" + gemm_csv_rel_perf_string + "\"\n}"
      gemm_csv_string = "function data_gemm() {\n return \"" + gemm_csv_string + "\"\n}"
      gemm_csv_max_perf_string = "function data_max_perf_gemm(){\n return \"" + gemm_csv_max_perf_string + "\"\n}"

      with open(ASSETS_DIR + "gemm_data_max_perf.js", "w+") as f:
        f.write(gemm_csv_max_perf_string)
      with open(ASSETS_DIR + "gemm_data.js", "w+") as f:
        f.write(gemm_csv_string)
      with open(ASSETS_DIR + "gemm_data_rel_perf.js", "w+") as f:
        f.write(gemm_csv_rel_perf_string)

      with open(ASSETS_DIR + "gemm_table_data_perf.html", "w+") as f:
        f.write(js_table_gemm_perf_string)
      with open(ASSETS_DIR + "gemm_table_data_rel_perf.html", "w+") as f:
        f.write(js_table_gemm_rel_perf_string)
      with open(ASSETS_DIR + "gemm_table_data_max_perf.html", "w+") as f:
        f.write(js_table_gemm_max_perf_string)

if __name__ == '__main__':
  main()
