#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
from datetime import datetime

CSM_PATH = '/home/scratch.svc_compute_arch/release/csm/latest/csm/csm'

def get_sec(time_str):
    # Get Seconds from time string
    if time_str == '':
        return 0
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# Example: wait_available_computelab_node.py --rule 'chip=ga100 and (node=computelab-*)'
def make_argparser():
    parser = argparse.ArgumentParser(description='Wait until node is avaliable')
    parser.add_argument('--rule', type=str, required=True,
                        default='', help='Rule for search node')
    parser.add_argument('--max-wait-time', type=str, required=False,
                        default='', help='max time in format HH:MM:SS to wait for node')
    return parser

if __name__ == '__main__':
    parser = make_argparser()
    args = make_argparser().parse_args()

    cmdline = CSM_PATH + ' find "' + args.rule + '" -f'
    timout_sec = get_sec(args.max_wait_time)
    begin = datetime.now()
    while True:
        print('Query computelab node, cmd = {}'.format(cmdline))
        cmd_ret = subprocess.run(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print('  stdout =  {}\n  stderr = {}'.format(cmd_ret.stdout.decode('utf-8').strip(), cmd_ret.stderr.decode('utf-8').strip()))
        if cmd_ret.stdout.decode('utf-8').strip().lower().find('error') >= 0:
            print('ERROR in csm!\n')
            break
        if len(cmd_ret.stdout.decode('utf-8').strip()) > 0:
            print('Find avaliable node!\n')
            exit(0)
        print("  No avaliable node.\n")
        end = datetime.now()
        second = (end-begin).total_seconds()
        if args.max_wait_time != '' and second > timout_sec:
            print("ERROR: Wait for node timeout, cannot submit job\n")
            break
        time.sleep(30)

    exit(1)


