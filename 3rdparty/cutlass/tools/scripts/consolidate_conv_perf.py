import sys
import csv
import os
import subprocess
import argparse
from conv_network import Resnet50, VGG_16, VNet, DarpaNet
import pandas as pd
import re, pdb

from collections import OrderedDict 
import numpy as np


csv_fieldnames = ['Network', 'DataTypeOp', 'DataTypeTag', 'chip', 'conv_kind', 'LayerID', 'Provider', \
                'OperationKind', 'TopOperation', 'BatchSize', 'LayerParams', 'stride', 'IteratorAlgorithm',\
                'split_k_mode', 'split_k_slices', 'Runtime', 'GFLOPs', 'Speedup', 'Utilization']

networks = ['Resnet50']
layers = {'Resnet50':range(1,20)}
providers = ['cudnn', 'cutlass']
data_type_tags = ['TensorOp(F16)', 'TensorOp(F32)', 'FFMA(F32)']
#data_type_tags = ['Inference(F16)[nhwc]', 'Inference(S8)[nhwc]', 'Inference(S8)[nc32hw32]', 'Inference(S4)[nhwc]', 'Inference(S4)[nc64hwc64]']
chips = ['ampere', 'volta', 'turing']
conv_kinds = ['fprop', 'dgrad', 'wgrad']
batch_sizes = ['408'] # training
#batch_sizes = ['34'] # inference


baseline_training = {'Provider':'cutlass', 'DataTypeTag':'FFMA(F32)'}
baseline_inference = {'Provider':'cudnn', 'DataTypeTag':'Inference(F16)[nhwc]'}

# Legend mapping
chip_map = {'ampere':'A100', 'volta':'TitanV', 'turing':'2080Ti'}
conv_kind_map = {'fprop':'Forward propagation', 'dgrad':'Backward data gradient', 'wgrad':'Backward weight gradient'}
provider_map = {'cudnn':'cuDNN', 'cutlass':'CUTLASS'}

# assemble all perf data into a dictionary
def AssembleTopOperations(args, reader, perf_data):

  for row in reader:
    #print(row)
    perf_data[(row['Network'],\
          row['DataTypeTag'],\
          row['chip'],\
          row['conv_kind'],\
          row['BatchSize'],\
          row['LayerID'],\
          row['Provider'].lower())] = row
  pass


# compute geo metric mean of an array
def GeoMean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

# create a geometric mean entry over all layers 
def GeoMeanCSVEntry(args, sample_row, data_series):
  sample_row['LayerID'] = 'GeoMean'
  sample_row['LayerParams'] = 'NA'

  # geo mean values
  sample_row['Speedup'] = GeoMean(data_series['Speedup'])
  sample_row['GFLOPs'] = GeoMean(data_series['GFLOPs'])
  sample_row['Runtime'] = GeoMean(data_series['Runtime'])

  return sample_row

# write all performance result and geo mean summary into a new csv file
def WriteSummary(args, perf_data, provider, network, data_type, chip, conv_kind, batch_size):
  print("WriteSummary")
  print(network, provider, data_type, chip, conv_kind, batch_size)
  file_exists = os.path.isfile(args.output)
  # open file in append mode
  with open(args.output, 'a', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=csv_fieldnames)

    if not file_exists:
      writer.writeheader()

    provider_series={'Speedup':[], 'GFLOPs':[], 'Runtime':[] }
    baseline_series={'Speedup':[], 'GFLOPs':[], 'Runtime':[] }


    for layer_id in layers[network]:

      provider_perf_key = (network, data_type, chip, conv_kind, batch_size, str(layer_id), provider)

      if args.runs == 'training':
        baseline_perf_key = (network, baseline_training['DataTypeTag'], chip, conv_kind, batch_size, str(layer_id), baseline_training['Provider'])
        #baseline_perf_key = (network, data_type, chip, conv_kind, batch_size, str(layer_id), baseline_training['Provider'])
      else:
        baseline_perf_key = (network, baseline_inference['DataTypeTag'], chip, conv_kind, batch_size, str(layer_id), baseline_inference['Provider'])
      
      provider_row = perf_data.get(provider_perf_key, None)
      
      if provider_row != None:
        # skip dgrad strided layers
        if provider_row['conv_kind'] == 'dgrad' and provider_row['stride'] == 'strided':
          continue

        # update speedup column for cutlass          
        baseline_row = perf_data.get(baseline_perf_key, None)
        if baseline_row != None:
          provider_row['Speedup'] = float(provider_row['GFLOPs'])/float(baseline_row['GFLOPs'])
          baseline_row['Speedup'] = float(baseline_row['GFLOPs'])/float(baseline_row['GFLOPs'])
          
          # write baseline row to summary .csv
          #print(baseline_row)
          writer.writerow(baseline_row)

          # save baseline entries in series
          baseline_series['Speedup'].append(float(baseline_row['Speedup']))
          baseline_series['GFLOPs'].append(float(baseline_row['GFLOPs']))
          baseline_series['Runtime'].append(float(baseline_row['Runtime']))

        # save entries in the series
        provider_series['Speedup'].append(float(provider_row['Speedup']))
        provider_series['GFLOPs'].append(float(provider_row['GFLOPs']))
        provider_series['Runtime'].append(float(provider_row['Runtime']))

        # write cutlass row to summary .csv
        #print(provider_row)
        writer.writerow(provider_row)

    # create geo-mean row for cutlass layers
    sample_row = perf_data.get((network, data_type, chip, conv_kind, batch_size, '1', provider), None)
    if sample_row != None:
      print(provider_series)
      provider_geo_mean_row = GeoMeanCSVEntry(args, sample_row, provider_series)
      writer.writerow(provider_geo_mean_row) 
    
    # create geo-mean row for baseline layers
    if args.runs == 'training' and data_type == baseline_training['DataTypeTag'] and provider == baseline_training['Provider']:
      sample_row = perf_data.get((network, data_type, chip, conv_kind, batch_size, '1', baseline_training['Provider']), None)
      #if sample_row != None:
      #  baseline_geo_mean_row = GeoMeanCSVEntry(args, sample_row, baseline_series)
      #  writer.writerow(baseline_geo_mean_row)
    

# iterate over perf for (data_type, chip, conv_kind, batch_size)
def SummarizeGTCPerformance(args, perf_data):

  for provider in providers: 
    for data_type in data_type_tags:
      for chip in chips:
        for conv_kind in conv_kinds:
          for batch_size in batch_sizes:
            # skip TensorOp(F32) from cudnn
            if provider == 'cudnn' and data_type == 'TensorOp(F32)':
              continue
            WriteSummary(args, perf_data, provider, 'Resnet50', data_type, chip, conv_kind, batch_size)

def ApplyLegendMapping(row):
  #row['chip'] = chip_map[row['chip']]
  #row['conv_kind'] = conv_kind_map[row['conv_kind']]
  #row['Provider'] = provider_map[row['Provider']]
  return row

# 
def RemoveDuplicatesUpdateLegends(args):
  filename, file_extension = os.path.splitext(args.output) 
  data_out = filename + '_no_duplicates' + file_extension
  with open(args.output, 'r', newline='') as in_file, open(data_out, 'w', newline='') as out_file:
    reader = csv.reader(in_file)
    writer = csv.writer(out_file)
    seen = set() # set for fast O(1) amortized lookup
    for row in reader:
        # apply legend mapping
        row = ApplyLegendMapping(row)
        row = tuple(row)
        if row in seen: continue # skip duplicate
        seen.add(row)
        writer.writerow(row)

# top-level function to go over all performance data in perf_data dict
def GTCPerformanceResults(args):
  perf_data = {}
  # recursively go over all .csv files and consolidate performance data
  for subdir, dirs, files in os.walk(args.base_dir):
    for filename in files:
        csv_filename = subdir + os.sep + filename
        print(csv_filename)
        with open(csv_filename, 'r') as csv_file:
          reader = csv.DictReader(csv_file)
          AssembleTopOperations(args, reader, perf_data)
          #print(len(perf_data))
          #print(perf_data)
  SummarizeGTCPerformance(args, perf_data)
  RemoveDuplicatesUpdateLegends(args)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--base-dir", default=".", help="Name of output file")
  parser.add_argument("--runs", default="training", help="'training' or 'inference' data type")
  parser.add_argument("--output", default="conv_network_top_operations.csv", help="Name of output file")
  parser.add_argument("--append", default="true", help="If true, final result file is opened for append.")

  args = parser.parse_args()

  GTCPerformanceResults(args)