'''spreadsheet.py extracts cutlass log data into CSV spreadsheet'''
import os
import random
import argparse
import re
import collections

try:
    import csv
except ImportError:
    csv = None

from helpers.utility         import OrderedDefaultDict
from helpers.Flags           import Flags
from helpers.cutlass_interface import OutputParser, LogExtractor, FileRange
from helpers.utility         import cutlass_flags_from_descs_str

import pdb

def print_step(message):
    print("\n")
    print("*******************************************************************************")
    print("* %s" % message)
    print("*******************************************************************************")

def make_help(s, has_choices=False):
    result = s + " [default: %(default)s]"

    if(has_choices):
        result += " [choices: %(choices)s]"

    return result

def spreadsheet_argparser():
    '''Generate parsed CLI argument for spreadsheet.py'''
    parser = argparse.ArgumentParser(description='cudnnTest spreadsheet extractor',
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-log_paths', '--log-paths', '-l',
            metavar = 'LOG_FILE',
            dest    = 'log_paths',
            default = None,
            action  = 'append',
            help    = make_help("Specify path(s) to load log from cudnn_run.py."))

    parser.add_argument('-spreadsheet', '--spreadsheet', '-o',
            metavar = 'CSV_FILE',
            dest    = 'spreadsheet',
            default = None,
            help    = make_help("Create perf spreadsheet with the given file name"))

    parser.add_argument('-layer_split_by_flag', '--layer-split-by-flag', '-s',
            metavar = 'FLAG',
            dest    = 'layer_split_by_flags',
            default = None,
            action  = 'append',
            help    = make_help("Split layers by given flag(s) (new row in \"spreadsheet\"); comma separate for multiple."))

    parser.add_argument('-extract', '--extract', '-e',
            metavar = 'EXTRACTOR_NAME',
            dest    = 'extract_value',
            default = None,
            action  = 'append',
            choices = LogExtractor.getExtratorList(),
            help    = make_help("For perf spreadsheet, specify extraction parameter; can specify multiple", True))

    return parser

def spreadsheet_post_argparse(parsed):
    '''Post process {parsed} command line argument:
    - Convert dest value to list of string if multiple values are allowed with comma separated value
    '''
    # -log_paths/--log-paths/-l:
    # Support both format -l A,B and -l A -l B, and remove duplicates using OrderedDict
    parsed.log_paths = collections.OrderedDict((f, None)
        for f_joined in parsed.log_paths
        for f in f_joined.split(',')).keys()
    if not parsed.log_paths:
        raise ValueError('missing --log-paths argument')
    # -spreadsheet/--spreadsheet/-k
    if parsed.spreadsheet is None:
        parsed.spreadsheet = os.path.splitext(parsed.log_paths[0])[0] + '.csv'
        if os.path.isfile(parsed.spreadsheet):
            print("WARNING: overwriting {}".format(parsed.spreadsheet))
    # post-process -layer_split_by_flag/--layer-split-by-flags/-s value
    # Support both -l flag1,flag2 and -s flag1 -s flag2; remove duplicates.
    if parsed.layer_split_by_flags is not None:
        parsed.layer_split_by_flags = collections.OrderedDict((f, None)
            for f_joined in parsed.layer_split_by_flags
            for f in f_joined.split(',')).keys()
    if parsed.extract_value is None:
        parsed.extract_value = ['time']
    return parsed


class CUTLASSDB(object):
    '''"Database" class for cutlass test results.
    from a list of cudnnTest logs, tracks list of CachedRunResult, namedtuple of file handle and start and stop position for test start and stop log patterns

    Corresponding logs between
    '''
    run_pattern = re.compile("\&\&\&\& RUNNING (.*)$")
    res_pattern = re.compile("\&\&\&\& FINISHED (.*)$")

    @staticmethod
    def iterLogEntries(file_desc, start_re, stop_re):
        """
        generates series of CachedRunResult instance whose field:
        .file_desc is {file_desc}
        .start     is the file position at the beginning of the line that matches {start_re}
        .stop      is the file position at the beginning of the line following the one that matches {end_re}
        """

    def __init__(self, file_path_list):
        self.file_desc_list = []
        self.cache_list = []
        self.cache_dict = collections.OrderedDict()
        for path in file_path_list:
            file_desc = open(path, 'r')
            self.file_desc_list.append(file_desc)

            for start_match, stop_match, cache in FileRange.fromMatches(file_desc, self.run_pattern, self.res_pattern, ret_matches=True):
                flags = start_match.group(1)
                self.cache_list.append(cache)
                self.cache_dict[flags] = cache # NOTE: only last one of given flags is kept.

    def __getitem__(self, key):
        if isinstance(key, int):
            item = self.cache_list[key] # raises IndexError
        elif isinstance(key, basestring):
            item = self.cache_dict[key] # raises KeyError
        else:
            raise KeyError('{!r} not found'.format(key))
        return item.read()

    def __iter__(self):
        for cache in self.cache_list:
            yield cache.read()

    def __del__(self):
        for fp in self.file_desc_list:
            fp.close()

def extract_result(output_parser, extract_value):
    if '.' in extract_value:
        name, field = extract_value.rsplit('.', 1)
    else:
        name, field = extract_value, None

    extracted = output_parser[name]

    if(extracted == None):
        return ""

    if field is None: # extract the first field
        return getattr(extracted, extracted._fields[0])
    else:
        return getattr(extracted, field)

def excel_stringify(string):
    return '"=""' + string + '"""'

def get_kernel_layout(kernel_name):
    layout_pattern = re.compile(r'_(nn|nt|tn|tt)')
    match = layout_pattern.search(kernel_name)
    if match:
        return match.groups(0)[0]
    else:
        raise Exception("Kernel name does not have a layout field") 

def generate_csv_table(cache_db, layer_split_by_flags=None, extract_value_list=['time']):
    '''Generate table and column from parsed spreadsheet.py CLI arguments'''
    # Contains index in table
    all_cols = collections.OrderedDict()

    table = OrderedDefaultDict(dict)

    split_flags = Flags()

    if layer_split_by_flags:
        for split_flag in layer_split_by_flags:
            split_flags[split_flag] = ('', )

    for output in cache_db:
        output_parser = OutputParser(output)

        if output_parser["layer_name"] == None:
            continue

        layer_name   = output_parser["layer_name"].layer_name
        test_flags   = cutlass_flags_from_descs_str(output_parser["test_flags"].test_flags)
        unique_flags = cutlass_flags_from_descs_str(output_parser["unique_flags"].unique_flags)

        if len(test_flags) != 1:
            raise Exception("Invalid test flags detected: %s" % test_flags)

        if len(unique_flags) != 1:
            raise Exception("Invalid unique flags detected: %s" % unique_flags)

        test_flags   = test_flags[0]
        unique_flags = unique_flags[0]
        unique_flags['kernel'] = (output_parser['function'].name,)

        current_split_flags = unique_flags.get_flags_for_keys(split_flags)

        base_flags = test_flags - (unique_flags - current_split_flags)

        #column = (unique_flags - split_flags).get_str()
        column = unique_flags['kernel'][0]
        row    = layer_name + current_split_flags.get_str(prefix='_', delimiter='')

        if row not in table:
            table[row] = OrderedDefaultDict(str)
            table[row]["Common Flags"] = base_flags.get_str()

        for extract_value in extract_value_list:
            value_col = '{}:{}'.format(extract_value, column)
            table[row][value_col] = extract_result(output_parser, extract_value)
            all_cols[value_col] = None
    return table, all_cols

def generate_csv_file(spreadsheet, table, all_cols):
    if csv is None:
        with open(spreadsheet, "w") as out_file:
            first_row = ["Layer Name", "Common Flags"] + [col for col in all_cols]

            out_file.write(",".join([excel_stringify(a) for a in first_row]) + '\n')

            for layer_name in table:
                row_data = table[layer_name]

                row = [excel_stringify(layer_name), excel_stringify(row_data["Common Flags"])] + [row_data[col] if col in row_data else '' for col in all_cols]

                out_file.write(",".join(row) + '\n')
    else: # use csv module
        csv_fields = ["Layer Name", "Common Flags"] + [col for col in all_cols]
        with open(spreadsheet, "w") as out_file:
            csv_writer = csv.DictWriter(out_file, csv_fields, dialect='excel', restval='', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()
            for layer_name, layer_data in table.iteritems():
                _row_dic = layer_data
                _row_dic['Layer Name'] = layer_name
                csv_writer.writerow(_row_dic)

def main():
    #*******************************************************************************
    #* Argument parsing logic
    #*******************************************************************************
    parsed_args = spreadsheet_argparser().parse_args()
    parsed_args = spreadsheet_post_argparse(parsed_args)

    #*******************************************************************************
    #* Print all parsed command line arguments
    #*******************************************************************************
    print_step("Printing all parsed command line arguments")
    for arg in parsed_args.__dict__:
        print("\t%s = %s" % (arg, parsed_args.__dict__[arg]))

    #*******************************************************************************
    #* Loading cache file
    #*******************************************************************************
    print_step("Loading cache file(s)")

    cache = CUTLASSDB(parsed_args.log_paths)

    #*******************************************************************************
    #* Generate table to be output in csv
    #*******************************************************************************
    print_step("Generating/Parsing output table")

    table, all_cols = generate_csv_table(cache, parsed_args.layer_split_by_flags, parsed_args.extract_value)

    #*******************************************************************************
    #* Output csv information
    #*******************************************************************************
    print_step("Outputting generated table to %s" % parsed_args.spreadsheet)

    generate_csv_file(parsed_args.spreadsheet, table, all_cols)

if __name__ == '__main__':
    main()

# eof
