from collections import namedtuple
from re          import compile, search
from utility     import is_ignored_line, flags_from_descs_str, cross_flags, get_flags_list_intersection, OrderedDefaultDict, flags_match_a_in_b_lists
from Flags       import Flags
from sys         import exc_info

import pdb

layer_pat = compile('\s*"(.*)"\s*=\s*(.*?)\s*$')
    
Layer      = namedtuple("Layer",    "base_name test_name test_diff_flags flags")

def gen_flags_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, label_db):
    with open(layer_file_name, "r") as layer_file:
        for (line_index, line) in enumerate(layer_file):
            # Ignore commented or empty lines
            if is_ignored_line(line):
                continue
            
            try:
                match_line = layer_pat.match(line)
                
                if match_line == None:
                    raise Exception("Invalid layer description given")
                                
                (layer_name, layer_descs) = match_line.groups()
                
                if search(whitelist_layer_name, layer_name) == None:
                    continue

                layer_descs_flags_list = flags_from_descs_str(layer_descs, label_db)

                if len(layer_descs_flags_list) == 0:
                    continue
                    
                layer_descs_flags_list = cross_flags(layer_descs_flags_list, global_flags_list)
                
                if len(layer_descs_flags_list) == 0:
                    continue
                
                filtered_layer_descs_flags_list = []
                
                for layer_descs_flags in layer_descs_flags_list:
                    for sub_flag in layer_descs_flags.get_sub_flags():
                        if not flags_match_a_in_b_lists(whitelist_flags_list, [sub_flag]):
                            continue
                    
                        filtered_layer_descs_flags_list.append(sub_flag)
                    
                if len(filtered_layer_descs_flags_list) == 0:
                    continue
                    
                unique_flags = get_flags_list_intersection(filtered_layer_descs_flags_list)
                
                for layer_descs_flags in filtered_layer_descs_flags_list:
                    yield (layer_name, unique_flags, layer_descs_flags)

                
            except Exception as e:
                # Store traceback info (to find where real error spawned)
                t, v, tb = exc_info()

                # Re-raise exception with line info
                raise t, Exception("[LAYER PARSING] %s (at %s:%s)" % (e.message, layer_file_name, line_index+1)), tb
    
    raise StopIteration
    
def get_layers_count_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, label_db):
    if split_flag_keys == None:
        split_flag_keys = []

    gen_layers = gen_flags_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, split_flag_keys, label_db)
    
    result_count = 0
    
    for (layer_name, unique_flags, flags) in gen_layers:
        result_count += flags.get_sub_flags_count()

    return result_count

def gen_layers_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, label_db):
    gen_layers = gen_flags_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, label_db)

    for (layer_name, unique_flags, flags) in gen_layers:

        for sub_flag in flags.get_sub_flags():
            test_diff_flags  = sub_flag.get_flags_for_keys(unique_flags)
                       
            test_name  = layer_name + test_diff_flags.get_str(prefix='_', delimiter='')
            
            # Uniquity test?
            yield Layer(layer_name, test_name, test_diff_flags, sub_flag)
            
    raise StopIteration
    
if __name__ == "__main__":
    test_str = '''"layer_name1" = n: 1
    
                  "layer_name2" = c: 2 * d: 4 * label_import'''
                  
    label_db = {"label_import": [Flags()]}
    
    print get_layers_from_str(test_str, label_db, "test_layer_name")
    
