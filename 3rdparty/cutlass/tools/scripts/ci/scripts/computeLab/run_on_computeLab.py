#! /usr/bin/env python3
#! python3

import os
import argparse
import subprocess
import re
import time
from datetime import datetime


# use case: run_on_computeLab.py --crun-args "-q "chip=tu104" -t 1:0:0 -b "/home/scratch.honghaol_gpu/workspace/gitlab_repo/xmma/scripts/linux/regression/run_and_process_sm70_smoke_tests.sh"" --time-wait-node 0:10:0
def make_argparser():
    parser = argparse.ArgumentParser(description="Wrapper of crun that can 1.wait until a node is avalible 2.block to monitor log and print stdout of task")
    parser.add_argument("--crun-args", type=str, required=True,
                        default="", help="arguments for crun")
    parser.add_argument("--time-wait-node", type=str, required=True,
                        default="", help="max time in format HH:MM:SS to wait for a node")
    parser.add_argument("--keyword-job-success", type=str, required=False,
                        default="[COMPUTELAB JOB SUCCESS]", help="keyword to indicate if job success")
    parser.add_argument("--time-monitor-job", type=str, required=False,
                        default="", help="max time in format HH:MM:SS to monitor a job log")
    return parser


def submit_crun_task(crun_args: str, wait_node_sec: int):
    # cmdline = "/home/scratch.svc_compute_arch/release/crun/latest/crun/crun " + crun_args.strip()
    cmdline = "/home/scratch.honghaol_gpu/workspace/program/crun_fork/0.1.200220183555/crun/crun " + crun_args.strip()
    print("Start to submit job to computeLab\n  cmd = {}\n".format(cmdline))
    begin = datetime.now()
    while True:
        # try submit task
        pty_m, pty_s = os.openpty()
        # to use crun on Jenkins pipelin, must pass stdin which is tty/pty along with environ
        current_env = os.environ.copy()
        print("env.LD_LIBRARY_PATH = {}".format(current_env["LD_LIBRARY_PATH"]))
        cmd_ret = subprocess.run(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=pty_s, shell=True, env=current_env)
        cmd_info_stdout = cmd_ret.stdout.decode("utf-8").strip()
        cmd_info_stderr = cmd_ret.stderr.decode("utf-8").strip()
        print("[stdout]:\n{}[stdout end]".format(cmd_info_stdout))
        print("[stderr]:\n{}[stderr end]".format(cmd_info_stderr))

        # if task submit success
        if cmd_info_stdout.find("Successfully submitted the job") >= 0 and cmd_info_stdout.find("Log file") >= 0:
            print("Job has been submitted to computeLab. ret1={} ret2={}".format(cmd_info_stdout.find("Successfully submitted the job"), cmd_info_stdout.find("Log file")))
            break

        # if wait enough time, exit
        end = datetime.now()
        second = (end-begin).total_seconds()
        if second > wait_node_sec:
            print("ERROR: Wait for node timeout, cannot submit job")
            exit(1)

        print("Cannot submit crun job, will try later...")
        time.sleep(600)

    # parse log file name and return
    log_filename = (re.findall(r"Log file +:(.+out)", cmd_info_stdout)[0]).strip()
    return log_filename


def monitor_crun_task(path_logfile: str, stop_keyword: str, success_keyword: str, monitor_job_sec: int):
    print("crun log file = '{}'".format(path_logfile))
    while not os.path.exists(path_logfile):  # In case file not exists
        print("Wait for log file to be created...")
        time.sleep(30)
    print("Start to monitor task log, stop_keyword = '{}', success_keyword = '{}'.".format(stop_keyword, success_keyword))
    ## Code below sometimes will get blocked when readline from pipe
    # f = subprocess.Popen(["tail", "-F", "-n", "+1", path_logfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # ret = 1
    # b_session_end = False
    # begin = datetime.now()
    # while not b_session_end:
    #     line = f.stdout.readline().decode("utf-8")  # will block to wait for a line
    #     print(line, end="")
    #     if line.find(success_keyword) >= 0:  # find success keyword, will return 1
    #         print("Find success keyword in print log.")
    #         ret = 0
    #     if line.find(stop_keyword) >= 0:  # crun session is end
    #         print("Find session end keyword, stop monitoring task log.")
    #         f.kill()
    #         b_session_end = True
    #     # if monitor enough time, exit
    #     end = datetime.now()
    #     second = (end-begin).total_seconds()
    #     if monitor_job_sec !=0 and second > monitor_job_sec:
    #         print("Timeout: stop monitoring task log.")
    #         ret = 1
    #         f.kill()
    #         b_session_end = True
    ret = 1
    begin = datetime.now()
    file = open(path_logfile, "r")
    while True:
        file_tell = file.tell()
        line = file.readline()
        if not line:
            file.close()
            time.sleep(30)
            file = open(path_logfile, "r")
            file.seek(file_tell)
        else:
            print(line, end="")
            if line.find(success_keyword) >= 0:  # find success keyword, will return 1
                print("Find success keyword in print log.")
                ret = 0
            if line.find(stop_keyword) >= 0:  # crun session is end
                print("Find session end keyword, stop monitoring task log.")
                break

        # if monitor enough time, exit
        end = datetime.now()
        second = (end-begin).total_seconds()
        if monitor_job_sec !=0 and second > monitor_job_sec:
            print("Timeout: stop monitoring task log.")
            break
    file.close()

    if ret != 0:
        # re-check whole file (for huge file, need to import mmap)
        with open(path_logfile, "r") as file:
            file_content = file.read()
            if success_keyword in file_content:
                print("Find success keyword in log file.")
                ret = 0

    return ret


def get_sec(time_str):
    # Get Seconds from time string
    if time_str == "":
        return 0
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


if __name__ == "__main__":
    # parse auguments
    parser = make_argparser()
    args = make_argparser().parse_args()

    # submit a job
    path_logfile = submit_crun_task(args.crun_args, get_sec(args.time_wait_node))
    # path_logfile = "/home/honghaol/slurm-279379.out"

    # monitor crun task output and print
    ret = monitor_crun_task(path_logfile, "session__end__date__", args.keyword_job_success, get_sec(args.time_monitor_job))

    exit(ret)
