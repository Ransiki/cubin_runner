import contextlib
import os.path
import time

from csv      import reader
from StringIO import StringIO
from re       import compile
from Flags    import Flags

import pdb

import subprocess

# Regex name match (capture a range of ASCII values)
re_name_match = "[\x21-\x7E]+"

empty_pat   = compile('\s*$')
comment_pat = compile('\s*//.*$')

def run_shell_command(cmd):
    return_code = None
    output      = None
    error_msg   = None
    try:
        # Start process with requested flags (and piped output from stdout & stderr)
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Obtain output from communicate()
        output, unused_err = process.communicate(None)

        # Poll for return code
        return_code = process.poll()

        # Error if return_code > 0
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd, output=output)

    except Exception as e:
        # Grab output if there is any
        if('output' in e.__dict__):
            output = e.output

        # Grab return code if there is any
        if('returncode' in e.__dict__):
            return_code = e.returncode

        # Grab error (guaranteed to be given)
        error_msg = str(e)

    return output, return_code, error_msg

def is_empty_line(line):
    return empty_pat.match(line) != None

def is_comment_line(line):
    return comment_pat.match(line) != None

def is_ignored_line(line):
    if is_empty_line(line) or is_comment_line(line):
        return True

    return False

def split_and_strip(string, split_by):
    return [val.strip() for val in string.split(split_by)]

def strip_exclude_quotes(string):
    result = ""

    in_quotes = False

    for char_idx, char in enumerate(string):
        if char == '"' and (char_idx == 0 or string[char_idx-1] != '\\'):
            in_quotes = not in_quotes

        if not char.isspace() or in_quotes:
            result += char

    return result


def split_comma(string):
    if string == None:
        return []

    stripped_string = strip_exclude_quotes(string)

    stripped_split = next(reader(StringIO(stripped_string)), [''])

    result = [val for val in stripped_split]

    return result

def split_space(string):
    if string == None:
        return []

    unstripped_result = next(reader(StringIO(string), delimiter=' '), [''])

    result = [val.strip() for val in unstripped_result if val.strip() != ""]

    return result

def get_shell_list(flags):
    return split_space(str(flags))

def cross_flags(flags_bases, flags_overrides):
    result = []

    for flags_base in flags_bases:
        for flags_override in flags_overrides:
            result.append(flags_base + flags_override)

    return result

## The below version fo flags_From_desc_str works when using spreadsheet.py with cutlass logs
def cutlass_flags_from_descs_str(string, label_db = None):
    if string == None:
        return [Flags(cutlass_flag=True)]

    # Case is assumed to be descriptors (and error if not)
    split_by_bar = split_and_strip(string, '*')

    flags = [Flags(cutlass_flag=True)]

    for val in split_by_bar:
        if '=' in val:
            (desc_key, desc_val) = split_and_strip(val, '=')

            for flags_idx in range(len(flags)):
                flags[flags_idx][desc_key.strip('--')] = tuple(split_comma(desc_val))

        elif is_empty_line(val):
            # Ignore empty areas between bars
            pass

        else:
            cross_label_name = val

            if not (cross_label_name in label_db):
                raise Exception("Label \"%s\" not found" % cross_label_name)

            flags = cross_flags(flags, label_db[cross_label_name])

    return flags

def flags_from_descs_str(string, label_db = None):
    if string == None:
        return [Flags()]

    # Case is assumed to be descriptors (and error if not)
    split_by_bar = split_and_strip(string, '*')

    flags = [Flags()]

    for val in split_by_bar:
        if ':' in val:
            (desc_key, desc_val) = split_and_strip(val, ':')

            for flags_idx in range(len(flags)):
                flags[flags_idx][desc_key] = tuple(split_comma(desc_val))

        elif is_empty_line(val):
            # Ignore empty areas between bars
            pass

        else:
            cross_label_name = val

            if not (cross_label_name in label_db):
                raise Exception("Label \"%s\" not found" % cross_label_name)

            flags = cross_flags(flags, label_db[cross_label_name])

    return flags

def flags_match_a_in_b(flags_a, flags_b):
    if flags_a.next_multi_flag():
        raise Exception("Error matching multi-flag a:" + repr(flags_a))

    if flags_b.next_multi_flag():
        raise Exception("Error matching multi-flag b:" + repr(flags_b))

    if flags_a.key_count() == 0:
        return True

    for flag_key in flags_a:
        if not (flag_key in flags_b):
            return False

        if flags_a[flag_key] != flags_b[flag_key]:
            return False

    return True

def flags_match_a_in_b_lists(flags_list_a, flags_list_b):
    for flags_a in flags_list_a:
        for flags_b in flags_list_b:
            for sub_flags_a in flags_a.get_sub_flags():
                for sub_flags_b in flags_b.get_sub_flags():
                    if flags_match_a_in_b(sub_flags_a, sub_flags_b):
                        return True


    return False

def get_flags_list_intersection(flags_list):

    if flags_list == None:
        raise Exception("None is not a valid flags list")

    if len(flags_list) == 0:
        raise Exception("Empty flags list given")

    base_flags  = dict( (key, flags_list[0][key]) for key in flags_list[0] )

    intersecting_flags = set([flag for flag in base_flags if len(base_flags[flag]) > 1])

    for flags in flags_list[1:]:
        for flag in base_flags:
            if flag not in flags or base_flags[flag] != flags[flag] or len(flags[flag]) > 1:
                intersecting_flags.add(flag)

        for flag in flags:
            if flag not in base_flags or flags[flag] != base_flags[flag] or len(base_flags[flag]) > 1:
                intersecting_flags.add(flag)

    return intersecting_flags


from collections import OrderedDict, defaultdict

class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

def format_time(seconds_float):
    ms_left = seconds_float * 1000

    day_left = int(int(ms_left) / (24*60*60*1000))
    ms_left -= day_left*(24*60*60*1000)

    hour_left = int(int(ms_left) / (60 * 60 * 1000))
    ms_left -= hour_left*(60*60*1000)

    min_left = int(int(ms_left) / (60*1000))
    ms_left -= min_left*(60*1000)

    sec_left = int(int(ms_left) / 1000)
    ms_left -= sec_left*(1000)

    result = ""
    if(day_left > 0):
        result += "%d days " % day_left

    if(hour_left > 0):
        result += "%d hours " % hour_left

    if(min_left > 0):
        result += "%d minutes " % min_left

    if(sec_left > 0):
        result += "%d seconds " % sec_left

    if(ms_left > 0):
        result += "%d milliseconds " % ms_left

    return result[:-1]

def set_default_dir(filename, dir_path, force=False):
    '''path join {dir_path}/{filename} if {filename} is simple basename, unless {force} is True'''
    if force or os.path.basename(filename) == filename:
        filename = os.path.join(dir_path, filename)
    return filename

def flag_to_dict(flag_str):
    '''Convert a flag string to dictionary
    {flag_str} is a '*' separated list of key-value pairs separated by ':', e.g.

    R:conv * formatIn:1 * formatOut:1 * dimA:"1,32,1,1"

    returns

        {
            'R'         : 'conv',
            'formatIn'  : '1',
            'formatOut' : '1',
            'dimA'      : ['1', '32', '1', '1']
        }

    - Does not perform type conversion to float/int, etc.
    - Does not current support direct product of flag values, e.g. R:conv,dgrad
    '''
    out = {}
    for s in flag_str.split('*'):
        s = s.strip().split(':', 1)
        if any(s):
            key, val = s
            if val.startswith('"') and val.endswith('"'):
                out[key] = val.strip('"').split(',') # split quoted list
            else:
                out[key] = val
    return out

@contextlib.contextmanager
def stopwatch(name):
    '''stopwatch to time a block of python code

        with stopwatch('title string'):
            {python code suite}

    will print

        "title string: 0.001s"

    after running {python code suite}
    '''
    t0 = time.time()
    try:
        yield t0
    finally:
        print("{}: {:g}s".format(name, time.time() - t0))


def prRed(skk): return ("\033[91m{}\033[00m" .format(skk)) 
def prGreen(skk): return ("\033[92m{}\033[00m" .format(skk)) 
def prYellow(skk): return ("\033[93m{}\033[00m" .format(skk)) 
def prLightPurple(skk): return ("\033[94m{}\033[00m" .format(skk)) 
def prPurple(skk): return ("\033[95m{}\033[00m" .format(skk)) 
def prCyan(skk): return ("\033[96m{}\033[00m" .format(skk)) 
def prLightGray(skk): return ("\033[97m{}\033[00m" .format(skk)) 
def prBlack(skk): return ("\033[98m{}\033[00m" .format(skk)) 


# eof
