import os
import random
import argparse
import re
import collections
import pdb

try:
    import csv
except ImportError:
    csv = None

from helpers.utility         import OrderedDefaultDict
from helpers.Flags           import Flags
from helpers.cutlass_interface import OutputParser, LogExtractor, FileRange
from helpers.utility         import cutlass_flags_from_descs_str
from spreadsheet             import CUTLASSDB, print_step, make_help, generate_csv_table


def analyze_argparser():
    '''Generate parsed CLI argument for spreadsheet.py'''
    parser = argparse.ArgumentParser(description='cudnnTest spreadsheet extractor',
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-log_paths', '--log-paths', '-l',
            metavar = 'LOG_FILE',
            dest    = 'log_paths',
            default = None,
            action  = 'append',
            help    = make_help("Specify path(s) to load log from cudnn_run.py."))

    parser.add_argument('-top_kernel', '--top-kernel', '-o',
            metavar = 'TOP_KERNEL',
            dest    = 'topkernel',
            default = False,
            help    = make_help("Finds top kernel and computes kernel strength"))


    parser.add_argument('-extract', '--extract', '-e',
            metavar = 'EXTRACTOR_NAME',
            dest    = 'extract_value',
            default = None,
            action  = 'append',
            choices = LogExtractor.getExtratorList(),
            help    = make_help("For perf spreadsheet, specify extraction parameter; can specify multiple", True))

    return parser

# creates spreadsheet of top kernel and computes kernel strength
def analyze_top_kernel(table, all_cols):

    kernel_name_time_dict = {} # {layer1: [(kernel_name, time),...], layer2: [(kernel_name, time),...], ...}

    for layer in table:
        for k in table[layer]:
            if k == 'Common Flags' or table[layer][k] == '0':
                continue
            kernel_name_pat = re.compile('time:--kernels=(.*)')
            kernel_name = ''
            match = kernel_name_pat.search(k)
            if match:
                kernel_name = match.groups(0)[0]
            else:
                raise Exception("Kernel name does not have a layout field") 

            if layer not in kernel_name_time_dict:
                kernel_name_time_dict[layer] = []
            kernel_name_time_dict[layer].append((kernel_name, float(table[layer][k])))

        # sort kernel_name_time_dict based on time element (second element). 
        # only add layers for which atleast one of the kernel is run  
        if layer in kernel_name_time_dict:
            kernel_name_time_dict[layer].sort(key=lambda x: x[1])

    # kernel_name_time_dict has (kernel_name, time) ins sorted order of time
    kernel_strength = {}
    f = open('top_kernels.csv', 'w')
    for layer in kernel_name_time_dict:
        f.write(layer)
        top_kernel = kernel_name_time_dict[layer][0][0]
        top_kernel_time = kernel_name_time_dict[layer][0][1]
        for elem in kernel_name_time_dict[layer]:
            val = float(top_kernel_time)/float(elem[1])
            f.write(','+elem[0]+' ('+str(val)+')')
            if elem[0] not in kernel_strength:
                kernel_strength[elem[0]] = []
            kernel_strength[elem[0]].append(float(top_kernel_time)/float(elem[1]))
        f.write('\n')
    f.close()
    
    # create kernel_strenght csv 
    fs = open('overall_kernels_strength.csv', 'w')
    
    for kernel in kernel_strength:
        strength = sum(kernel_strength[kernel])/len(kernel_strength[kernel])
        fs.write(kernel+','+str(strength)+'\n')
    
    fs.close()
    
'''
#*******************************************************************************
#* Analyze top kernel for each layer
#*******************************************************************************
kernel_name_time_dict = {}
for layer in table:
    for k in table[layer]:
        if 'base_flags' in k or '-Dheurgen_dbg=-1' in k:
            continue
        if layer not in kernel_name_time_dict:
            kernel_name_time_dict[layer] = []
        kernel_name_time_dict[layer].append((table[layer][k].split(':')[0], table[layer][k].split(':')[1]))

    # sort kernel_name_time_dict[layer] based on second element of the tuple    
    if layer in kernel_name_time_dict:
        kernel_name_time_dict[layer].sort(key=lambda x: x[1])


# kernel_name_time_dict has (kernel_name, time) ins sorted order of time
kernel_strength = {}
f = open('top_kernels.csv', 'w')
for layer in kernel_name_time_dict:
    f.write(layer)
    top_kernel = kernel_name_time_dict[layer][0][0]
    top_kernel_time = kernel_name_time_dict[layer][0][1]
    for elem in kernel_name_time_dict[layer]:
        val = float(top_kernel_time)/float(elem[1])
        f.write(','+elem[0]+' ('+str(val)+')')
        if elem[0] not in kernel_strength:
            kernel_strength[elem[0]] = []
        kernel_strength[elem[0]].append(float(top_kernel_time)/float(elem[1]))
    f.write('\n')
f.close()

# create kernel_strenght csv 
fs = open('overall_kernels_strength.csv', 'w')

for kernel in kernel_strength:
    strength = sum(kernel_strength[kernel])/len(kernel_strength[kernel])
    fs.write(kernel+','+str(strength)+'\n')

fs.close()
'''

def main():
    #*******************************************************************************
    #* Argument parsing logic
    #*******************************************************************************
    parsed_args = analyze_argparser().parse_args()

    #*******************************************************************************
    #* Print all parsed command line arguments
    #*******************************************************************************
    print_step("Printing all parsed command line arguments")
    for arg in parsed_args.__dict__:
        print("\t%s = %s" % (arg, parsed_args.__dict__[arg]))

    #*******************************************************************************
    #* Loading cache file
    #*******************************************************************************
    print_step("Loading cutlass perf test results from file(s)")

    cache = CUTLASSDB(parsed_args.log_paths)

    #*******************************************************************************
    #* Generate table for analysis
    #*******************************************************************************
    print_step("Generating/Parsing output table")

    table, all_cols = generate_csv_table(cache)

    if parsed_args.topkernel:
        analyze_top_kernel(table, all_cols)

if __name__ == '__main__':
    main()


# eof

