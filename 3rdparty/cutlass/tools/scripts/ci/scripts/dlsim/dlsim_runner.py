# External import
import os
import csv
import re
import argparse
import platform
import subprocess
import shutil
from pathlib import Path
from objdict import ObjDict
from collections import defaultdict

# Internal import
from dlsim.dlsim_translator import translate
from dlsim.utils import correlate_dlsim
from dlsim import PerfList
from run_presilicon_performance_tests import PERF_DB_NETWORK_PERFSIM_NAME


def initialize(workspace, cleanup):
    # Only support running DLSim on Linux at the moment
    assert platform.system != 'Linux', "Linux is the only tested system that can run DLSim"

    # FIXME: determine Linux distribution

    for file in get_dlsim_outfile(workspace):
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaning existing file {file}")

    if cleanup:
        try:
            dlsim = os.path.join(workspace, "dlsim")
            print(f"Cleaning existing DLSim project {dlsim}")
            shutil.rmtree(dlsim)
        except FileNotFoundError:
            pass
    
    if not os.path.exists(workspace):
        print(f"Creating working directory: {workspace}")
        Path(workspace).mkdir(parents=True, exist_ok=True)
    

def install_dlsim(workspace, build_dlsim):
    install_script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'dlsim_install.sh'
    )

    build_dlsim = "-b" if build_dlsim else ""

    subprocess.check_call(
        ['bash', install_script_path, "-w", workspace, build_dlsim], 
        cwd=workspace
    )


def parent_dir(current, level):
    """
    Go number of 'level' up from 'folder' 
    """
    if level == 0:
        return current
    
    return parent_dir(os.path.dirname(current), level - 1)


def get_perflist(workspace, perf_list):
    if perf_list == "perf_smart":
        mode = "smart"
    elif perf_list == "perf_perfsim":
        mode = "perfsim"
    else:
        raise Exception(f"perf list {perf_list} is not supported")

    input_csv = os.path.join(workspace, f"FK_perf_{mode}_testlist_SM100_cutlass3x_gemm.csv")
    assert os.path.exists(input_csv), f"{input_csv} does not exist"

    return input_csv


def get_dlsim_infile(workspace):
    file_name = "gemm.csv"
    input_csv = os.path.join(workspace, file_name)

    return input_csv, file_name


def get_dlsim_outfile(workspace):
    file_name = get_dlsim_infile(workspace)[1]

    result_csv = os.path.join(workspace, f"rslt_{file_name}")
    error_csv = os.path.join(workspace, f"err_{file_name}")

    return result_csv, error_csv


def construct_dlsim_workload(workspace, perf_list, cc, pi_path):
    # Use generator.py to generate perf testlist
    cuda_version_major = "12"
    architectures = re.sub(r'[^0-9]', '', cc)  # e.g. 100
    kernels = f"*{cc}*"  # e.g. sm100

    if perf_list == "perf_smart":
        test_set_name = "kernel_perflist_smart"
    elif perf_list == "perf_perfsim":
        test_set_name = "kernel_perflist_perfsim"
    else:
        raise Exception(f"perf list {perf_list} is not supported")


    generator_script = os.path.join(
        parent_dir(os.path.realpath(__file__), 6),
        "python",
        "cutlass_library",
        "generator.py"
    )
    subprocess.check_call(
        [
            "python3",
            generator_script,
            "--generator-target", test_set_name,
            "--cuda-version", cuda_version_major,
            "--architectures", architectures,
            "--kernels", kernels,
            "--disable-cutlass-package-imports"
        ],
        cwd=workspace
    )

    translate(ObjDict({
        "perf_list": perf_list,
        "input_file": get_perflist(workspace, perf_list),
        "output_file": get_dlsim_infile(workspace)[0],
        "cc": cc,
        "pi_path": pi_path
    }))


def run_dlsim(workspace, pi_path):
    virtual_env = os.path.join(workspace, "dlsim", "python", "venv", "bin", "python3")
    run_sript_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dlsim_gemm_runner.sh')
    _, in_file = get_dlsim_infile(workspace)
    dlsim_cmd = ""

    try:
        cmd = [
            "bash", run_sript_path, 
            "-v", virtual_env, 
            "-i", in_file]

        if pi_path is not None:
            cmd += ["-p", pi_path]

        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE)
        
        # Parse stdout to get dlsim cmd
        while True:
            line = proc.stdout.readline().decode("utf-8") 
            if not line:
                break
            if line.startswith("dlsim cmd:") and not dlsim_cmd:
                dlsim_cmd = str(line.lstrip("dlsim cmd:").rstrip("\n"))
            print(line)

    except subprocess.CalledProcessError as e:
        print(f"Non-zero return code {e.returncode} when running dlsim. output: {e.output}")
        raise e

    return dlsim_cmd

def dlsim_perfdb_key(result):
    return result["run_tag"]


def post_process(workspace, dlsim_cmd):
    result_csv, error_csv = get_dlsim_outfile(workspace)

    if os.path.exists(error_csv):
        print(f"Error happens when running dlsim, check {error_csv} for details")

    if not os.path.exists(result_csv):
        raise Exception(f"dlsim result csv does not exists under {result_csv}")
    
    dlsim_result = defaultdict(dict)

    with open(result_csv) as result:
        csvreader = csv.DictReader(result)
        for row in csvreader:
            dlsim_result[dlsim_perfdb_key(row)] = dict(row)
            dlsim_result[dlsim_perfdb_key(row)]["dlsim_cmd"] = dlsim_cmd
    
    return dlsim_result


def update_dlsim_records(workspace, perf_list, dlsim_result, real_run):
    if perf_list == "perf_perfsim":
        correlate_dlsim(workspace, dlsim_result, PERF_DB_NETWORK_PERFSIM_NAME, real_run)
    elif perf_list == "perf_smart":
        pass
    else:
        raise Exception(f"perf list {perf_list} is not supported")


def run(args):
    assert args.cc != "sm100" "Only sm100 dlsim is integrated for cutlass perf testing, please add support for other gpus if needed"

    # Init environment
    initialize(args.workspace, args.cleanup)

    # Install DLSim package locally
    install_dlsim(args.workspace, args.build_dlsim)

    # Generate DLSim input file
    construct_dlsim_workload(args.workspace, args.perf_list, args.cc, args.pi_path)

    # Run DLSim
    dmsim_cmd = run_dlsim(args.workspace, args.pi_path)

    # Results post process
    dlsim_result = post_process(args.workspace, dmsim_cmd)

    # Push results to perf database
    update_dlsim_records(args.workspace, args.perf_list, dlsim_result, args.real_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DLSim runner for cutlass smart and perfsim testlist')
    parser.add_argument("--perf_list", default="perf_perfsim", type=PerfList, choices=list(PerfList), help="perf_perfsim or perf_smart")
    parser.add_argument("--workspace", default="/tmp/workspace", help="Absolute path to the base directory where we run DLSim")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup existing dlsim project if there is one")
    parser.add_argument("--build_dlsim", action="store_true", help="Build dlsim from source")
    parser.add_argument("--real_run", action="store_true", help="Default is dry_run, when real_run, perf database will be refreshed")
    parser.add_argument("--cc", default="sm100", help="target GPU's cc")
    parser.add_argument('--pi_path', default=None, help='path to dump dlsim PI report data')
    args = parser.parse_args()

    dlsim_runner_args = ObjDict({
        "perf_list": args.perf_list.value,
        "workspace": args.workspace,
        "cleanup": bool(args.cleanup),
        "build_dlsim": bool(args.build_dlsim),
        "real_run": bool(args.real_run),
        "cc": args.cc,
        "pi_path": args.pi_path
    })

    print(dlsim_runner_args)

    run(dlsim_runner_args)
